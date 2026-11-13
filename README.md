# PyTorch DTW C++ extension

Dynamic time warping in native PyTorch, with CPU and CUDA backends.

```bash
pip install torchdtw
```

This package requires PyTorch 2.10 or later. It is developed using the PyTorch
2.10 Stable ABI, and compiled with instructions for CUDA cards from Volta to Blackwell.
It is available on Linux (with CUDA support), macOS, and Windows (without CUDA).
This was originally made for [fastabx](https://github.com/bootphon/fastabx), but
it can be used in other projects. Only the exact DTW is implemented, there is
no plan to add variants.

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
dtw_batch(distances, sx=None, sy=None, *, symmetric=False)
```

Compute the batched DTW cost on the ``distances`` 4D tensor.

Only the ``(sx[i], sy[j])`` sub-block of each pair is read, so padding beyond the sequence
lengths is ignored. Every ``sx[i]`` must be ``<= s1`` and every ``sy[j] <= s2``: the CPU backend
raises an error, and the CUDA backend a device-side assert, if violated. Use ``+inf``
to mask forbidden alignments. NaN distances are unsupported: the result is unspecified and may
differ between the CPU and CUDA backends. Integer ``distances`` accumulate the cost in their own
dtype and may overflow on long sequences; use a wide enough integer dtype or a floating dtype.

**Parameters:**

- **distances** (<code>Tensor</code>) – A 4D tensor of shape (n1, n2, s1, s2) representing the pairwise distances between two
batches of sequences.
- **sx** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n1,) with the lengths of the sequences in the first batch.
Defaults to s1.
- **sy** (<code>Tensor | None</code>) – An optional 1D integer tensor of shape (n2,) with the lengths of the sequences in the second batch.
Defaults to s2.
- **symmetric** (<code>bool</code>) – Whether the two batches are the same: only the pairs ``i < j`` are computed, and mirrored,
and the diagonal is 0. Requires ``n1 == n2``.

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


<!-- /griffe -->

## Performance

For many DTWs on short sequences, prefer `dtw_batch` over a Python loop of `dtw` calls.
A single `dtw_batch` launches one CUDA kernel (one block per pair) or one parallel CPU
loop, amortizing dispatch, allocation, and launch overhead across the whole batch.

On CUDA, each pair is computed by one block, which keeps three anti-diagonals of the
dynamic program in shared memory when they fit: when `distances.size(-1)` is at most 2730
for `dtw` and `dtw_batch` in float32, float16 and bfloat16 (1638 in float64).
Beyond, they are kept in global memory, with no limit on the sequence lengths other than
`s1 + s2 - 1 <= 65535` for `dtw` and `dtw_batch`.

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
