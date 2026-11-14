# PyTorch DTW C++ extension

Dynamic time warping (DTW) and soft-DTW in native PyTorch, with CPU and CUDA backends.

```bash
pip install torchdtw
```

Two variants are implemented: the exact DTW (`dtw`, `dtw_batch`, `dtw_path`), and
soft-DTW (`soft_dtw`, `soft_dtw_batch`), its differentiable relaxation
(Cuturi & Blondel, 2017) to use as a loss.

This package requires PyTorch 2.10 or later. It is developed using the PyTorch
2.10 Stable ABI, and compiled with instructions for CUDA cards from Volta to Blackwell.
It is available on Linux (with CUDA support), macOS, and Windows (without CUDA).
This was originally made for [fastabx](https://github.com/bootphon/fastabx), but
it can be used in other projects.

<!-- griffe -->
## Usage

This package provides:

### `dtw`

```python
dtw(distances, *, normalize=True)
```

Compute the DTW cost of the given ``distances`` 2D tensor.

Use ``+inf`` to mask forbidden alignments. NaN distances are unsupported: the result is
unspecified and may differ between the CPU and CUDA backends.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 2D tensor of shape (n, m) with the pairwise distances between two sequences.
- **normalize** (<code>bool</code>) – Whether to divide the cost by the length of the warping path.

**Returns:**

- <code>Tensor</code> – A scalar tensor with the cost.

### `dtw_batch`

```python
dtw_batch(distances, sx=None, sy=None, *, symmetric=False, normalize=True)
```

Compute the DTW cost between all pairs of sequences of two batches, on the ``distances`` 4D tensor.

Only the ``(sx[i], sy[j])`` sub-block of each pair is read, so the padding beyond the sequence lengths is ignored.
Use ``+inf`` to mask forbidden alignments. NaN distances are unsupported: the result is unspecified and may
differ between the CPU and CUDA backends.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 4D tensor of shape (n1, n2, s1, s2) with the pairwise distances between the frames of two
batches of sequences.
- **sx** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n1,) with the lengths of the first batch, at most s1.
Defaults to s1.
- **sy** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n2,) with the lengths of the second batch, at most s2.
Defaults to s2.
- **symmetric** (<code>bool</code>) – Whether the two batches are the same: only the pairs ``i < j`` are computed, and mirrored,
and the diagonal is 0, the cost of a sequence with itself when the distance of a frame to itself is 0.
Requires ``n1 == n2``.
- **normalize** (<code>bool</code>) – Whether to divide each cost by the length of its warping path.

**Returns:**

- <code>Tensor</code> – A 2D tensor of shape (n1, n2) with the costs.

### `dtw_path`

```python
dtw_path(distances)
```

Compute the DTW path of the given ``distances`` 2D tensor.

No CUDA variant or batched implementation are provided for now. Use ``+inf`` to mask forbidden alignments.
NaN distances are unsupported and give an unspecified path.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 2D tensor of shape (n, m) with the pairwise distances between two sequences.

**Returns:**

- <code>Tensor</code> – A 2D tensor of shape (*, 2) with the indices of the path, from (0, 0) to (n - 1, m - 1).

### `soft_dtw`

```python
soft_dtw(distances, sx=None, sy=None, *, gamma=1.0, bandwidth=-1, normalize=False)
```

Compute the soft-DTW cost of a pair of sequences, or of each pair in a batch of independent pairs.

Soft-DTW replaces the minimum over alignments of DTW by the soft minimum
``-gamma * log(sum(exp(-cost / gamma)))``, so it is differentiable with respect to ``distances``. Unlike
``dtw``, the cost is not normalized by the path length by default. With 3D ``distances``, only the
``(sx[b], sy[b])`` sub-block of each pair is read.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 2D tensor of shape (N, M) with the pairwise distances between the frames of two sequences,
or a 3D tensor of shape (B, N, M) for B independent pairs, e.g. ``torch.cdist(x, y) ** 2`` for sequences x
of shape (B, N, D) and y of shape (B, M, D). Must have a floating dtype.
- **sx** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (B,) with the lengths of the first sequences, at most N.
Defaults to N. Only with 3D ``distances``.
- **sy** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (B,) with the lengths of the second sequences, at most M.
Defaults to M. Only with 3D ``distances``.
- **gamma** (<code>float</code>) – The smoothing parameter, positive.
- **bandwidth** (<code>int</code>) – Width of the Sakoe-Chiba band around the diagonal of each pair, in steps along the longer
sequence. Negative values disable the band.
- **normalize** (<code>bool</code>) – Whether to divide the cost of each pair by ``sx[b] + sy[b] - 1``, the length of its longest
alignment path.

**Returns:**

- <code>Tensor</code> – A scalar tensor with the cost for 2D ``distances``, a 1D tensor of shape (B,) with the costs for 3D
ones.

### `soft_dtw_batch`

```python
soft_dtw_batch(distances, sx=None, sy=None, *, gamma=1.0, bandwidth=-1, symmetric=False, normalize=False)
```

Compute the soft-DTW cost between all pairs of sequences of two batches, on the ``distances`` 4D tensor.

Same as ``soft_dtw``, for every pair ``(i, j)`` instead of B independent pairs. Only the ``(sx[i], sy[j])``
sub-block of each pair is read.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 4D tensor of shape (n1, n2, s1, s2) with the pairwise distances between the frames of two
batches of sequences. Must have a floating dtype.
- **sx** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n1,) with the lengths of the first batch, at most s1.
Defaults to s1.
- **sy** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n2,) with the lengths of the second batch, at most s2.
Defaults to s2.
- **gamma** (<code>float</code>) – The smoothing parameter, positive.
- **bandwidth** (<code>int</code>) – Width of the Sakoe-Chiba band, as in ``soft_dtw``. Negative values disable the band.
- **symmetric** (<code>bool</code>) – Whether the two batches are the same: only the pairs ``i <= j`` are computed, from the blocks
``distances[i, j]``, and mirrored. Unlike in ``dtw_batch``, the diagonal is computed. Requires ``n1 == n2``.
- **normalize** (<code>bool</code>) – Whether to divide the cost of each pair by ``sx[i] + sy[j] - 1``, as in ``soft_dtw``.

**Returns:**

- <code>Tensor</code> – A 2D tensor of shape (n1, n2) with the costs.


<!-- /griffe -->

## Soft-DTW

Like the exact DTW, soft-DTW takes the distances between the frames of the sequences,
so any distance works. For example, with the squared Euclidean distance:

```python
loss = soft_dtw(torch.cdist(x, y) ** 2, sx, sy, gamma=0.1).mean()
```

The soft-DTW kernels started as [torchsoftdtw](https://github.com/angelo-ortiz/torchsoftdtw).

## Benchmark

Check [this folder](https://github.com/mxmpl/torchdtw/tree/main/benchmark) for comparisons
against reference implementations.

## Building from source

To build the CUDA backend from source, CUDA must be found by PyTorch (for example
with `CUDA_HOME`). The GPU architectures to compile for are set by
`TORCH_CUDA_ARCH_LIST`. If it is not set, the build targets the architectures of the
visible GPUs only, and fails if no GPU is visible. Set it to build for other GPUs:

```bash
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX" pip install --no-build-isolation torchdtw
```

### ROCm backend

The published wheels do not include a ROCm backend. To get one, build from source
on Linux in an environment with a ROCm build of PyTorch, setting the AMD GPU
architectures you target:

```bash
PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx1100" pip install --no-build-isolation torchdtw
```

The ROCm backend is built from the same CUDA kernel and gives the same results and
error checks as the CUDA backend.

## Performance

For many DTWs on short sequences, prefer `dtw_batch` over a Python loop of `dtw` calls.
A single `dtw_batch` launches one CUDA kernel (one block per pair) or one parallel CPU
loop, amortizing dispatch, allocation, and launch overhead across the whole batch.

On CUDA, each pair is computed by one block, which keeps three anti-diagonals of the
dynamic program in shared memory when they fit: when `distances.size(-1)` is at most 2730
for `dtw` and `dtw_batch` in float32, float16, bfloat16, uint8, int8 and int16 (1638 in
float64, int32 and int64), and 4096 for `soft_dtw` and `soft_dtw_batch` (2048 in float64).
Beyond, they are kept in global memory, with no limit on the sequence lengths other than
`sx[i] + sy[j] - 1 <= 65535` for each pair.

The forward pass of soft-DTW only keeps the matrix its backward pass needs, of the
size of `distances`, when `distances` requires grad. Under `torch.no_grad()`, or for
inputs that do not require grad, it uses no memory beyond its output.

## Edge cases

The CPU and CUDA backends give bit-identical results, except for NaN distances, which are unsupported: their
result is unspecified and may differ between the backends.
Ties between alignments break as diagonal <= left <= up.

### Dtypes

| `distances` dtype | `dtw`, `dtw_batch`, `dtw_path` | `soft_dtw`, `soft_dtw_batch` |
|---|---|---|
| float32, float64 | Accumulated in the input dtype. | Accumulated in the input dtype. |
| float16, bfloat16 | Accumulated in float32, cast back to the input dtype: an unnormalized cost beyond its range is `inf`. | Same. |
| uint8, int8, int16 | Accumulated in int32, or in int64 on the CPU when the longest path could overflow int32. | Rejected. |
| int32, int64 | Accumulated in int64: int64 costs are exact. | Rejected. |
| bool, complex | Rejected. | Rejected. |

With integer distances:

- The normalized cost is divided in int64, rounded toward zero.
- The unnormalized cost is cast to the input dtype, and wraps:
  `dtw(torch.full((1, 3), 100, dtype=torch.int8), normalize=False)` is 44.

`sx` and `sy` can have any integer dtype (uint8, int8, int16, int32, int64, uint16, uint32, uint64), the same
for both, and must be on the device of `distances`.

### Sizes and lengths

| Case | CPU | CUDA |
|---|---|---|
| 2D `dtw` or `soft_dtw` with `n == 0` or `m == 0` | Empty pair: cost 0, as in the batch ops. | Same. |
| `dtw_path` with `n == 0` or `m == 0` | Empty path, of shape (0, 2). | Same. |
| Empty batch: `n1 == 0`, `n2 == 0` or `B == 0` | Empty output. | Empty output. |
| `s1 == 0` or `s2 == 0`, or a length <= 0 | Empty pair: cost 0, and gradient 0 for soft-DTW. | Same. |
| A length larger than the padded size (`sx[i] > s1`) | Raises. | Device-side assert. |
| A uint64 length >= 2^63 | Raises. | Device-side assert. |
| A pair whose longest path has more than 65535 cells (`n + m - 1 > 65535` for `dtw`, `sx[i] + sy[j] - 1 > 65535` for the others) | No limit. | Raises when neither `sx` nor `sy` is given (checked on the padded sizes); device-side assert otherwise. |
| `n1` or `n2` above 2^31 - 1, or more than 2^31 - 1 pairs with `symmetric` | No limit. | Raises. |

### Values

- Use `+inf` to mask forbidden alignments. If every alignment of a pair goes through a masked cell, its cost is
  `+inf`, for DTW and for soft-DTW.
- With `symmetric`, `dtw_batch` does not read the blocks `distances[i, j]` with `i >= j`: its diagonal is 0.
  `soft_dtw_batch` does not read the blocks with `i > j`, and computes the diagonal, since the soft-DTW cost of a
  sequence with itself is not 0.
- Soft-DTW tends to the unnormalized DTW cost as `gamma` tends to 0.
- The Sakoe-Chiba band of soft-DTW is floored at half the slope of the diagonal joining the first and last frames
  of each pair, so that an alignment always exists, even with `bandwidth=0`.

### Gradients

- `dtw`, `dtw_batch` and `dtw_path` are not differentiable: no gradient flows back to `distances`. Use soft-DTW for
  gradients.
- Soft-DTW gives no gradient (zeros) to the padding beyond `sx` and `sy`, to empty pairs, and with `symmetric` to
  the blocks it does not read.
- Soft-DTW is differentiable once: its backward pass is not differentiable, so second-order derivatives through it
  are not available.

### Devices

- `dtw_path` only has a CPU kernel: a CUDA input is copied to the host, and the path back to the device.
- On Windows, torchdtw is built without CUDA support, and CUDA inputs raise.
- The ROCm backend has the same edge cases as the CUDA backend.

## Citation

Please cite the fastabx paper if you use this package in your work:

```bib
@misc{fastabx,
  title={fastabx: A library for efficient computation of ABX discriminability},
  author={Maxime Poli and Emmanuel Chemla and Emmanuel Dupoux},
  year={2025},
  eprint={2505.02692},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.02692},
}
```
