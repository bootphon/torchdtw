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

- **distances** (<code>Tensor</code>) – A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.
- **normalize** (<code>bool</code>) – Whether to divide the cost by the length of the warping path.

**Returns:**

- <code>Tensor</code> – A scalar tensor with the cost.

### `dtw_batch`

```python
dtw_batch(distances, sx=None, sy=None, *, symmetric=False, normalize=True)
```

Compute the batched DTW cost on the ``distances`` 4D tensor.

Only the ``(sx[i], sy[j])`` sub-block of each pair is read, so padding beyond the sequence
lengths is ignored. Every ``sx[i]`` must be ``<= s1`` and every ``sy[j] <= s2`. Use ``+inf``
to mask forbidden alignments. NaN distances are unsupported: the result is unspecified and may
differ between the CPU and CUDA backends.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 4D tensor of shape (n1, n2, s1, s2) representing the pairwise distances between two
batches of sequences.
- **sx** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n1,) with the lengths of the sequences in the first batch.
Defaults to s1.
- **sy** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n2,) with the lengths of the sequences in the second batch.
Defaults to s2.
- **symmetric** (<code>bool</code>) – Whether the two batches are the same: only the pairs ``i < j`` are computed, and mirrored,
and the diagonal is 0. Requires ``n1 == n2``.
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

- **distances** (<code>Tensor</code>) – A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.

**Returns:**

- <code>Tensor</code> – A 2D tensor of shape (*, 2) with the path indices.

### `soft_dtw`

```python
soft_dtw(distances, sx=None, sy=None, *, gamma=1.0, bandwidth=-1, normalize=False)
```

Compute the soft-DTW cost of a pair of sequences, or of each pair in a batch of independent pairs.

Soft-DTW replaces the minimum over alignments of DTW by the soft minimum
``-gamma * log(sum(exp(-cost / gamma)))``, so it is differentiable with respect to ``distances``. Unlike
``dtw``, the cost is not normalized by the path length. With 3D ``distances``, only the ``(sx[b], sy[b])``
sub-block of each pair is read, and the padding beyond it gets no gradient. Every ``sx[b]`` must be ``<= N``
and every ``sy[b] <= M``.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 2D tensor of shape (N, M) with the pairwise distances between the frames of two sequences,
or a 3D tensor of shape (B, N, M) for B independent pairs, e.g. ``torch.cdist(x, y) ** 2`` for sequences x
of shape (B, N, D) and y of shape (B, M, D). Must have a floating dtype.
- **sx** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (B,) with the lengths of the first sequences. Defaults to N.
Only with 3D ``distances``.
- **sy** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (B,) with the lengths of the second sequences. Defaults to M.
Only with 3D ``distances``.
- **gamma** (<code>float</code>) – The smoothing parameter, positive. Soft-DTW tends to the unnormalized DTW cost as it tends to 0.
- **bandwidth** (<code>int</code>) – Width of the Sakoe-Chiba band around the diagonal joining the first and last frames of each
pair, in steps along the longer sequence. It is floored at half the slope of that diagonal so that an
alignment always exists. Negative values disable the band.
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
sub-block of each pair is read, and the padding beyond it gets no gradient. Every ``sx[i]`` must be ``<= s1``
and every ``sy[j] <= s2``.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 4D tensor of shape (n1, n2, s1, s2) with the pairwise distances between the frames of two
batches of sequences. Must have a floating dtype.
- **sx** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n1,) with the lengths of the first batch. Defaults to s1.
- **sy** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n2,) with the lengths of the second batch. Defaults to s2.
- **gamma** (<code>float</code>) – The smoothing parameter, positive.
- **bandwidth** (<code>int</code>) – Width of the Sakoe-Chiba band, as in ``soft_dtw``. Negative values disable the band.
- **symmetric** (<code>bool</code>) – Whether the two batches are the same: only the pairs ``i <= j`` are computed, from the blocks
``distances[i, j]``, and mirrored. The blocks ``i > j`` are not read and get no gradient. Unlike in
``dtw_batch``, the diagonal is computed: the soft-DTW cost of a sequence with itself is not 0. Requires
``n1 == n2``.
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
