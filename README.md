# PyTorch DTW C++ extension

Dynamic time warping in native PyTorch, with CPU and CUDA backends.

```bash
pip install torchdtw
```

This package requires PyTorch 2.10 or later. It is developed using the PyTorch
2.10 Stable ABI, and compiled with instructions for CUDA cards from Volta to Blackwell.
This was originally made for [fastabx](https://github.com/bootphon/fastabx), but
it can be used in other projects. Only the exact DTW is implemented, there is
no plan to add variants.

## Usage
 
This package provides three functions:

### DTW

```python
def dtw(distances: torch.Tensor) -> torch.Tensor
```

Compute the DTW cost of the given ``distances`` 2D tensor.

**Arguments**:

- `distances`: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.

**Returns**:

A scalar tensor with the cost.

### DTW path

```python
def dtw_path(distances: torch.Tensor) -> torch.Tensor
```

Compute the DTW path of the given ``distances`` 2D tensor.

No CUDA variant or batched implementation are provided for now.

**Arguments**:

- `distances`: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.

**Returns**:

A 2D tensor of shape (*, 2) with the path indices.

### Batched DTW

```python
def dtw_batch(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *,
              symmetric: bool) -> torch.Tensor
```

Compute the batched DTW cost on the ``distances`` 4D tensor.

**Arguments**:

- `distances`: A 4D tensor of shape (n1, n2, s1, s2) representing the pairwise distances between two
batches of sequences.
- `sx`: A 1D tensor of shape (n1,) representing the lengths of the sequences in the first batch.
- `sy`: A 1D tensor of shape (n2,) representing the lengths of the sequences in the second batch.
- `symmetric`: Whether or not the DTW is symmetric (i.e., the two batches are the same).

**Returns**:

A 2D tensor of shape (n1, n2) with the costs.

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
