# PyTorch DTW C++ extension

Dynamic time warping (DTW) and soft-DTW in native PyTorch, with CPU and CUDA backends.

```bash
pip install torchdtw
```

This package requires PyTorch 2.10 or later. It is developed using the PyTorch
2.10 Stable ABI, and compiled with instructions for CUDA cards from Volta to Blackwell.
It is available on Linux (with CUDA support), macOS, and Windows (without CUDA).
This was originally made for [fastabx](https://github.com/bootphon/fastabx), but
it can be used in other projects. Two variants are implemented: the exact DTW
(`dtw`, `dtw_batch`, `dtw_path`), normalized by the length of the optimal path, and
soft-DTW (`soft_dtw`, `soft_dtw_batch`), its differentiable relaxation
(Cuturi & Blondel, 2017), to use as a loss.

<!-- griffe -->
## Usage

This package provides:

### `dtw`

```python
dtw(distances)
```

Compute the DTW cost of the given ``distances`` 2D tensor.

Use ``+inf`` to mask forbidden alignments. NaN distances are unsupported: the result is
unspecified and may differ between the CPU and CUDA backends. Integer ``distances`` accumulate
the cost in their own dtype and may overflow on long sequences; use a wide enough integer dtype
or a floating dtype.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.

**Returns:**

- <code>Tensor</code> – A scalar tensor with the cost.

### `dtw_batch`

```python
dtw_batch(distances, sx, sy, *, symmetric)
```

Compute the batched DTW cost on the ``distances`` 4D tensor.

Only the ``(sx[i], sy[j])`` sub-block of each pair is read, so padding beyond the sequence
lengths is ignored. Every ``sx[i]`` must be ``<= s1`` and every ``sy[j] <= s2``: the CPU backend
validates this, but the CUDA backend assumes it and reads out of bounds if violated. Use ``+inf``
to mask forbidden alignments. NaN distances are unsupported: the result is unspecified and may
differ between the CPU and CUDA backends. Integer ``distances`` accumulate the cost in their own
dtype and may overflow on long sequences; use a wide enough integer dtype or a floating dtype.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 4D tensor of shape (n1, n2, s1, s2) representing the pairwise distances between two
batches of sequences.
- **sx** (<code>Tensor</code>) – A 1D tensor of shape (n1,) representing the lengths of the sequences in the first batch.
- **sy** (<code>Tensor</code>) – A 1D tensor of shape (n2,) representing the lengths of the sequences in the second batch.
- **symmetric** (<code>bool</code>) – Whether or not the DTW is symmetric (i.e., the two batches are the same).

**Returns:**

- <code>Tensor</code> – A 2D tensor of shape (n1, n2) with the costs.

### `dtw_path`

```python
dtw_path(distances)
```

Compute the DTW path of the given ``distances`` 2D tensor.

No CUDA variant or batched implementation are provided for now.
Use ``+inf`` to mask forbidden alignments. NaN distances are unsupported and give an
unspecified path.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.

**Returns:**

- <code>Tensor</code> – A 2D tensor of shape (*, 2) with the path indices.

### `soft_dtw`

```python
soft_dtw(distances, sx=None, sy=None, *, gamma=1.0, bandwidth=-1)
```

Compute the soft-DTW cost of each pair of sequences in the ``distances`` 3D tensor.

Soft-DTW replaces the minimum over alignments of DTW by the soft minimum
``-gamma * log(sum(exp(-cost / gamma)))``, so it is differentiable with respect to ``distances``. Unlike
``dtw``, the cost is not normalized by the path length. Only the ``(sx[b], sy[b])`` sub-block of each pair is
read, and the padding beyond it gets no gradient. Every ``sx[b]`` must be ``<= N`` and every ``sy[b] <= M``:
the CPU backend validates this, but the CUDA backend assumes it and reads out of bounds if violated.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 3D tensor of shape (B, N, M) with the pairwise distances between the frames of B pairs of
sequences, e.g. ``torch.cdist(x, y) ** 2`` for sequences x of shape (B, N, D) and y of shape (B, M, D).
Must have a floating dtype.
- **sx** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (B,) with the lengths of the first sequences. Defaults to N.
- **sy** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (B,) with the lengths of the second sequences. Defaults to M.
- **gamma** (<code>float</code>) – The smoothing parameter, positive. Soft-DTW tends to the unnormalized DTW cost as it tends to 0.
- **bandwidth** (<code>int</code>) – Width of the Sakoe-Chiba band around the diagonal joining the first and last frames of each
pair, in steps along the longer sequence. It is floored at half the slope of that diagonal so that an
alignment always exists. Negative values disable the band.

**Returns:**

- <code>Tensor</code> – A 1D tensor of shape (B,) with the costs.

### `soft_dtw_batch`

```python
soft_dtw_batch(distances, sx=None, sy=None, *, gamma=1.0, bandwidth=-1)
```

Compute the soft-DTW cost between all pairs of sequences of two batches, on the ``distances`` 4D tensor.

Same as ``soft_dtw``, for every pair ``(i, j)`` instead of B independent pairs. Only the ``(sx[i], sy[j])``
sub-block of each pair is read, and the padding beyond it gets no gradient. Every ``sx[i]`` must be ``<= s1``
and every ``sy[j] <= s2``: the CPU backend validates this, but the CUDA backend assumes it.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 4D tensor of shape (n1, n2, s1, s2) with the pairwise distances between the frames of two
batches of sequences. Must have a floating dtype.
- **sx** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n1,) with the lengths of the first batch. Defaults to s1.
- **sy** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n2,) with the lengths of the second batch. Defaults to s2.
- **gamma** (<code>float</code>) – The smoothing parameter, positive.
- **bandwidth** (<code>int</code>) – Width of the Sakoe-Chiba band, as in ``soft_dtw``. Negative values disable the band.

**Returns:**

- <code>Tensor</code> – A 2D tensor of shape (n1, n2) with the costs.


<!-- /griffe -->

## Performance

For many DTWs on short sequences, prefer `dtw_batch` over a Python loop of `dtw` calls.
A single `dtw_batch` launches one CUDA kernel (one block per pair) or one parallel CPU
loop, amortizing dispatch, allocation, and launch overhead across the whole batch.
The same holds for `soft_dtw` (B independent pairs) and `soft_dtw_batch` (all pairs
between two batches).

On CUDA, each pair is computed by one block, which keeps three anti-diagonals of the
dynamic program in shared memory when they fit: when `distances.size(-1)` is at most 2730
for `dtw` and `dtw_batch` in float32, float16 and bfloat16 (1638 in float64), and 4096 for
soft-DTW (2048 in float64).
Beyond, they are kept in global memory, with no limit on the sequence lengths other than
`s1 + s2 - 1 <= 65535` for `dtw` and `dtw_batch`.

The forward pass of soft-DTW only keeps the matrix its backward pass needs, of the
size of `distances`, when `distances` requires grad. Under `torch.no_grad()`, or for
inputs that do not require grad, it uses no memory beyond its output.

## Soft-DTW

Like the exact DTW, soft-DTW takes the distances between the frames of the sequences,
so any distance works. For example, with the squared Euclidean distance, as a loss
normalized by the length of the longest alignment:

```python
cost = soft_dtw(torch.cdist(x, y) ** 2, sx, sy, gamma=0.1)
loss = (cost / (sx + sy - 1)).mean()
```

The soft-DTW kernels started as [torchsoftdtw](https://github.com/angelo-ortiz/torchsoftdtw),
which builds on ideas from [pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda)
and [sdtw-cuda-torch](https://github.com/BGU-CS-VIL/sdtw-cuda-torch).

## Benchmark

Check [this folder](https://github.com/mxmpl/torchdtw/tree/main/benchmark) for comparisons
against reference implementations.

## ROCm backend

The published wheels do not include a ROCm backend. To get one, build from source
on Linux in an environment with a ROCm build of PyTorch, setting the AMD GPU
architectures you target:

```bash
PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx1100" pip install --no-build-isolation torchdtw
```

The ROCm backend is built from the same CUDA kernel and gives the same results and
error checks as the CUDA backend.

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
