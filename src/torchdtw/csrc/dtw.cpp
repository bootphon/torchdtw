#include <Python.h>

#include "dtw.h"
#include <algorithm>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>
#include <vector>

// Creates a dummy empty _C module that can be imported from Python.
PyMODINIT_FUNC PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}

namespace torchdtw {

template <typename scalar_t> struct HardCell {
  using acc = acc_t<scalar_t>;
  PairDistances<scalar_t> distances;

  acc origin() const { return distances(0, 0); }
  acc edge(int64_t i, int64_t j, acc parent) const { return distances(i, j) + parent; }
  acc inner(int64_t i, int64_t j, acc diag, acc left, acc up) const {
    return distances(i, j) + std::min({up, diag, left});
  }
};

template <typename scalar_t>
static void
compute_dtw_cost(const TensorAccessor<const scalar_t, 2>& distances_a, int64_t N, int64_t M, acc_t<scalar_t>* cost) {
  fill_grid(HardCell<scalar_t>{PairDistances<scalar_t>(distances_a)}, N, M, cost, M);
}

template <typename cost_t, typename Visit>
static void trace_dtw_path(const cost_t* cost, int64_t N, int64_t M, Visit&& visit) {
  int64_t i = N - 1;
  int64_t j = M - 1;
  visit(i, j);
  while (i > 0 || j > 0) {
    if (i == 0) {
      j--;
    } else if (j == 0) {
      i--;
    } else {
      const cost_t c_up = cost[(i - 1) * M + j];
      const cost_t c_left = cost[i * M + j - 1];
      const cost_t c_diag = cost[(i - 1) * M + j - 1];
      if (c_diag <= c_left && c_diag <= c_up) {
        i--;
        j--;
      } else if (c_left <= c_up) {
        j--;
      } else {
        i--;
      }
    }
    visit(i, j);
  }
}

template <typename scalar_t>
static scalar_t compute_dtw_value(
    const TensorAccessor<const scalar_t, 2>& distances_a, int64_t N, int64_t M,
    std::vector<acc_t<scalar_t>>& workspace) {
  if (static_cast<int64_t>(workspace.size()) < N * M) {
    workspace.resize(N * M);
  }
  compute_dtw_cost<scalar_t>(distances_a, N, M, workspace.data());
  int64_t path_len = 0;
  trace_dtw_path(workspace.data(), N, M, [&](int64_t, int64_t) { path_len++; });
  return normalized_cost<scalar_t>(workspace[N * M - 1], path_len);
}

Tensor dtw_cpu(const Tensor& distances) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  Tensor out = torch::stable::new_empty(distances, torch::headeronly::IntHeaderOnlyArrayRef());
  dispatch_dtw(distances.scalar_type(), [&]<typename scalar_t>() {
    thread_local std::vector<acc_t<scalar_t>> workspace;
    const scalar_t result = compute_dtw_value<scalar_t>(accessor<const scalar_t, 2>(distances), N, M, workspace);
    torch::stable::fill_(out, result);
  });
  return out;
}

Tensor dtw_path_cpu(const Tensor& distances) {
  STD_TORCH_CHECK(distances.dim() == 2, "distances must be a 2D tensor");
  const int64_t N = distances.size(0);
  const int64_t M = distances.size(1);
  STD_TORCH_CHECK(N > 0 && M > 0, "Empty input tensor");
  std::vector<std::pair<int64_t, int64_t>> path;
  dispatch_dtw(distances.scalar_type(), [&]<typename scalar_t>() {
    std::vector<acc_t<scalar_t>> cost(N * M);
    compute_dtw_cost<scalar_t>(accessor<const scalar_t, 2>(distances), N, M, cost.data());
    path.reserve(N + M - 1);
    trace_dtw_path(cost.data(), N, M, [&](int64_t i, int64_t j) { path.push_back({i, j}); });
  });
  std::reverse(path.begin(), path.end());
  Tensor out = torch::stable::new_empty(distances, {(int64_t)path.size(), 2}, torch::headeronly::ScalarType::Long);
  std::memcpy(
      reinterpret_cast<int64_t*>(out.data_ptr()),
      reinterpret_cast<const int64_t*>(path.data()),
      static_cast<size_t>(path.size() * 2) * sizeof(int64_t));
  return out;
}

template <typename distances_t>
void dtw_batch_cpu_impl(
    Tensor& out, const Tensor& distances, torch::headeronly::IntHeaderOnlyArrayRef sizes, const Lengths sx,
    const Lengths sy, bool symmetric) {
  using acc = acc_t<distances_t>;
  check_lengths(sx, sy, sizes[0], sizes[1], sizes[2], sizes[3]);
  const auto distances_a = accessor<const distances_t, 4>(distances);
  auto out_a = accessor<distances_t, 2>(out);
  parallel_pairs<acc>(sizes[0], sizes[1], symmetric, false, [&](int64_t x, int64_t y, std::vector<acc>& workspace) {
    const int64_t N = sx[x];
    const int64_t M = sy[y];
    const distances_t value =
        N > 0 && M > 0 ? compute_dtw_value<distances_t>(distances_a[x][y], N, M, workspace) : distances_t(0);
    out_a[x][y] = value;
    if (symmetric)
      out_a[y][x] = value;
  });
}

static Tensor dtw_batch_cpu(const Tensor& distances, const Tensor* sx, const Tensor* sy, bool symmetric) {
  return dtw_batch_op(
      distances,
      sx,
      sy,
      symmetric,
      [&]<typename distances_t>(
          Tensor& out, torch::headeronly::IntHeaderOnlyArrayRef sizes, const Lengths& sx_l, const Lengths& sy_l) {
        dtw_batch_cpu_impl<distances_t>(out, distances, sizes, sx_l, sy_l, symmetric);
      });
}

Tensor dtw_batch_lengths_cpu(const Tensor& distances, const Tensor& sx, const Tensor& sy, bool symmetric) {
  return dtw_batch_cpu(distances, &sx, &sy, symmetric);
}

Tensor dtw_batch_optional_lengths_cpu(
    const Tensor& distances, const std::optional<Tensor>& sx, const std::optional<Tensor>& sy, bool symmetric) {
  return dtw_batch_cpu(distances, optional_ptr(sx), optional_ptr(sy), symmetric);
}

STABLE_TORCH_LIBRARY(torchdtw, m) {
  m.def("dtw(Tensor distances) -> Tensor");
  m.def("dtw_path(Tensor distances) -> Tensor");
  m.def("dtw_batch(Tensor distances, Tensor sx, Tensor sy, bool symmetric) -> Tensor");
  m.def("dtw_batch.optional_lengths(Tensor distances, Tensor? sx, Tensor? sy, bool symmetric) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchdtw, CPU, m) {
  m.impl("dtw", &TORCH_BOX(dtw_cpu));
  m.impl("dtw_path", &TORCH_BOX(dtw_path_cpu));
  m.impl("dtw_batch", &TORCH_BOX(dtw_batch_lengths_cpu));
  m.impl("dtw_batch.optional_lengths", &TORCH_BOX(dtw_batch_optional_lengths_cpu));
}

} // namespace torchdtw
