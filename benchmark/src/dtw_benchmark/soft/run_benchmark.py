"""Soft-DTW benchmark: compare torchdtw against the other soft-DTW implementations, forward and backward.

Each batch shape runs at three spreads of sequence lengths: "none" (every sequence has the padded length), "small"
(lengths down to 90% of it) and "large" (down to 25%). torchdtw is the correctness reference of every case.
"""

import warnings
from pathlib import Path

import torch
from numba.core.errors import NumbaPerformanceWarning
from torch.utils.benchmark import Compare, Measurement, Timer

from dtw_benchmark.run_benchmark import _inject

from .implementations import available

GAMMA = 1.0
SEED = 0
# (batch size, padded length, frame dimension): from many short pairs to few long ones.
CONFIGS = [(64, 16, 8), (64, 32, 8), (64, 64, 8), (64, 256, 12), (32, 256, 16), (16, 512, 16)]
# Smallest length of a spread, as a fraction of the padded length.
SPREADS = {"none": 1.0, "small": 0.9, "large": 0.25}
BENCHMARK_MARKERS = ("<!-- benchmark-soft -->", "<!-- /benchmark-soft -->")


def make_batch(
    batch_size: int, length: int, dim: int, spread: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random padded sequences ``(x, y)`` and their lengths ``(sx, sy)``, with zeroed padding."""
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(batch_size, length, dim, generator=g)
    y = torch.randn(batch_size, length, dim, generator=g)
    lengths = []
    for seq in (x, y):
        low = max(1, round(SPREADS[spread] * length))
        seq_lengths = torch.randint(low, length + 1, (batch_size,), generator=g)
        for b in range(batch_size):
            seq[b, seq_lengths[b] :] = 0
        lengths.append(seq_lengths)
    return x.to(device), y.to(device), lengths[0].to(device), lengths[1].to(device)


def measurements(
    config: tuple[int, int, int], spread: str, device: torch.device, *, backward: bool, min_run_time: float
) -> list[Measurement]:
    """Time every implementation on one configuration, after checking it against torchdtw."""
    x, y, sx, sy = make_batch(*config, spread, device)
    impls = available(device)

    # Warm up (Numba JIT compilation, allocator caching), then check correctness.
    outputs = {}
    for name, fn in impls.items():
        for _ in range(2):
            outputs[name] = fn(x, y, sx, sy, GAMMA)
    for name, out in outputs.items():
        torch.testing.assert_close(out.float(), outputs["torchdtw"], atol=1e-3, rtol=1e-3, msg=name)

    x = x.clone().requires_grad_(backward)

    def run(fn: object) -> None:
        cost = fn(x, y, sx, sy, GAMMA)
        if backward:
            x.grad = None
            cost.sum().backward()

    batch_size, length, dim = config
    return [
        Timer(
            stmt="run(fn)",
            globals={"run": run, "fn": fn},
            num_threads=torch.get_num_threads(),
            label=f"{device.type}{' (forward + backward)' if backward else ' (forward)'}",
            sub_label=name,
            description=f"B={batch_size} L={length} D={dim} {spread}",
        ).blocked_autorange(min_run_time=min_run_time)
        for name, fn in impls.items()
    ]


def benchmark(min_run_time: float = 0.2, output: Path | None = None, *, backward: bool = True) -> None:
    """Benchmark soft-DTW. With ``output``, inject the results into a markdown file."""
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    devices = [torch.device(t) for t in ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])]
    results = [
        m
        for device in devices
        for config in CONFIGS
        for spread in SPREADS
        for m in measurements(config, spread, device, backward=backward, min_run_time=min_run_time)
    ]
    compare = Compare(results)
    compare.colorize()
    compare.print()
    if output is not None:
        _inject(output, results, BENCHMARK_MARKERS)
