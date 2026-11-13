"""The benchmark cases and their inputs, shared by ``run`` and ``compare``.

Torch-only, so that ``compare_release.py``, run as a script, can import it as a sibling module: ``run`` and
``compare`` then measure the same cases.
"""

import torch

SEED = 0
# Single-pair shapes (N, M): square ones plus a couple of rectangular (N != M) ones.
SHAPES = [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (128, 512), (512, 128)]
# Batch configs (n1, n2, s1, s2, symmetric): mostly the many-small-pairs workload torchdtw targets, where the per-call
# overhead and the split of the pairs between CPU threads dominate, plus a few large pairs.
BATCH_CONFIGS = [
    (8, 8, 32, 32, True),
    (8, 8, 32, 32, False),
    (16, 16, 8, 8, True),
    (16, 16, 8, 8, False),
    (16, 16, 32, 32, True),
    (16, 16, 32, 32, False),
    (32, 32, 32, 32, True),
    (32, 32, 32, 32, False),
    (64, 64, 16, 16, True),
    (128, 128, 16, 16, True),
    (128, 128, 16, 16, False),
    (256, 256, 8, 8, True),
    (16, 16, 128, 128, True),
    (4, 8, 256, 512, False),
]
# Spreads of the sequence lengths of a batch: the smallest length, as a fraction of the padded length. "none": every
# sequence has the padded length, "small": lengths down to 90% of it, "large": down to 25%.
SPREADS = {"none": 1.0, "small": 0.9, "large": 0.25}
# The dtype of the lengths of the main workload (fastabx). ``compare`` also measures int64.
LENGTHS_DTYPE = torch.int32


def devices() -> list[torch.device]:
    """The CPU, then CUDA when available."""
    return [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])


def shape_label(shape: tuple[int, int]) -> str:
    """Label of a single-pair case."""
    return f"{shape[0]}x{shape[1]}"


def batch_label(config: tuple[int, int, int, int, bool], spread: str) -> str:
    """Label of a batch case."""
    n1, n2, s1, s2, symmetric = config
    return f"n={n1}x{n2} s={s1}x{s2} {'sym' if symmetric else 'asym'} {spread}"


def random_distances(shape: tuple[int, ...], *, generator: torch.Generator) -> torch.Tensor:
    """Uniform random float32 distances in [0, 1)."""
    return torch.rand(shape, generator=generator, dtype=torch.float32)


def symmetric_distances(n: int, s: int, *, generator: torch.Generator) -> torch.Tensor:
    """Random ``(n, n, s, s)`` distances of a batch with itself: ``d[j, i]`` is the transpose of ``d[i, j]``."""
    d = random_distances((n, n, s, s), generator=generator)
    lower = torch.ones(n, n, dtype=torch.bool).tril(-1)
    d[lower] = d.transpose(0, 1).mT[lower]
    return d


def lengths(n: int, s: int, spread: str, *, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    """Random lengths of ``n`` sequences padded to ``s``, uniform between the smallest length of ``spread`` and ``s``."""
    low = max(1, round(SPREADS[spread] * s))
    return torch.randint(low, s + 1, (n,), generator=generator, dtype=dtype)


def single_input(shape: tuple[int, int], *, device: torch.device, seed: int = SEED) -> torch.Tensor:
    """Distances of a single-pair case."""
    return random_distances(shape, generator=torch.Generator().manual_seed(seed)).to(device)


def batch_inputs(
    config: tuple[int, int, int, int, bool],
    spread: str,
    *,
    device: torch.device,
    seed: int = SEED,
    lengths_dtype: torch.dtype = LENGTHS_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distances and lengths ``(d, sx, sy)`` of a batch case. A symmetric batch has ``sy == sx``."""
    n1, n2, s1, s2, symmetric = config
    g = torch.Generator().manual_seed(seed)
    if symmetric:
        if n1 != n2 or s1 != s2:
            msg = f"A symmetric batch needs n1 == n2 and s1 == s2, got {config}"
            raise ValueError(msg)
        d = symmetric_distances(n1, s1, generator=g)
        sx = lengths(n1, s1, spread, generator=g, dtype=lengths_dtype)
        sy = sx.clone()
    else:
        d = random_distances((n1, n2, s1, s2), generator=g)
        sx = lengths(n1, s1, spread, generator=g, dtype=lengths_dtype)
        sy = lengths(n2, s2, spread, generator=g, dtype=lengths_dtype)
    return d.to(device), sx.to(device), sy.to(device)
