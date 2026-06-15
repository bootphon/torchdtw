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

This package provides three functions:

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


<!-- /griffe -->

## Performance

For many DTWs on short sequences, prefer `dtw_batch` over a Python loop of `dtw` calls.
A single `dtw_batch` launches one CUDA kernel (one block per pair) or one parallel CPU
loop, amortizing dispatch, allocation, and launch overhead across the whole batch.

## Benchmark

Check [this folder](https://github.com/mxmpl/torchdtw/tree/main/benchmark) for comparisons
against reference implementations.

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
