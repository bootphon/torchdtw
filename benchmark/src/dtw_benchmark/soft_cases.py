"""The soft-DTW benchmark cases and their inputs, shared by ``run soft`` and ``compare``.

Torch-only, like ``cases``, so that ``compare_release.py``, run as a script, can import it as a sibling module. The
spreads and the dtype of the lengths are those of ``cases``; the all-pairs cases of ``soft_dtw_batch`` are the batch
configs of ``cases``.
"""

from collections.abc import Callable

import torch

if __package__:
    from . import cases
else:  # imported by ``compare_release.py`` run as a script
    import cases  # ty: ignore[import-not-found]

GAMMA = 1.0
CONFIGS = [(64, 16, 8), (64, 32, 8), (64, 64, 8), (64, 256, 12), (32, 256, 16), (16, 512, 16)]


def config_label(config: tuple[int, int, int], spread: str) -> str:
    """Label of a paired case."""
    batch_size, length, dim = config
    return f"B={batch_size} L={length} D={dim} {spread}"


def sequences(
    config: tuple[int, int, int],
    spread: str,
    *,
    device: torch.device,
    seed: int = cases.SEED,
    lengths_dtype: torch.dtype = cases.LENGTHS_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random padded sequences ``(x, y)`` of a paired case, of shape (B, L, D), and their lengths ``(sx, sy)``.

    The padding beyond the lengths is zero.
    """
    batch_size, length, dim = config
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, length, dim, generator=g)
    y = torch.randn(batch_size, length, dim, generator=g)
    sx = cases.lengths(batch_size, length, spread, generator=g, dtype=lengths_dtype)
    sy = cases.lengths(batch_size, length, spread, generator=g, dtype=lengths_dtype)
    frames = torch.arange(length)
    x[frames >= sx[:, None].long()] = 0
    y[frames >= sy[:, None].long()] = 0
    return x.to(device), y.to(device), sx.to(device), sy.to(device)


def paired_inputs(
    config: tuple[int, int, int],
    spread: str,
    *,
    device: torch.device,
    seed: int = cases.SEED,
    lengths_dtype: torch.dtype = cases.LENGTHS_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distances and lengths ``(d, sx, sy)`` of a paired case: squared Euclidean distances, of shape (B, L, L)."""
    x, y, sx, sy = sequences(config, spread, device=device, seed=seed, lengths_dtype=lengths_dtype)
    return torch.cdist(x, y) ** 2, sx, sy


def gradient(fn: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    """Gradient of the sum of ``fn(inputs)`` with respect to ``inputs``: a forward and a backward pass."""
    leaf = inputs.detach().requires_grad_()
    (grad,) = torch.autograd.grad(fn(leaf).sum(), leaf)
    return grad
