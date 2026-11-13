"""DTW benchmark: compare torchdtw's ``dtw`` and ``dtw_batch`` against the other DTW implementations.

Each batch config runs at every spread of sequence lengths of ``cases.SPREADS``, with int32 lengths. torchdtw is
the correctness reference of every case. Tables have one row per case and one column per implementation.
"""

import functools
from pathlib import Path

import torch
from torch.utils.benchmark import Measurement

from dtw_benchmark import cases, report, timing

from .implementations import NAIVE, REFERENCE, available

MARKERS = ("<!-- benchmark -->", "<!-- /benchmark -->")


def single_measurements(shape: tuple[int, int], device: torch.device, *, min_run_time: float) -> list[Measurement]:
    """Time every single-pair implementation on one shape, after checking it against torchdtw."""
    x = cases.single_input(shape, device=device)
    impls = available(device, batched=False)
    if shape != cases.SHAPES[0]:
        del impls[NAIVE]  # O(N*M) Python loop: only on the smallest shape
    calls = {name: functools.partial(fn, x) for name, fn in impls.items()}
    timing.warm_up_and_check(calls, REFERENCE)
    return [
        timing.wall_time(
            call,
            label=f"dtw / {device.type}",
            sub_label=cases.shape_label(shape),
            description=name,
            min_run_time=min_run_time,
            num_threads=torch.get_num_threads(),
        )
        for name, call in calls.items()
    ]


def batch_measurements(
    config: tuple[int, int, int, int, bool], spread: str, device: torch.device, *, min_run_time: float
) -> list[Measurement]:
    """Time every batched implementation on one config and spread, after checking it against torchdtw."""
    d, sx, sy = cases.batch_inputs(config, spread, device=device)
    symmetric = config[-1]
    calls = {
        name: functools.partial(fn, d, sx, sy, symmetric=symmetric)
        for name, fn in available(device, batched=True).items()
    }
    timing.warm_up_and_check(calls, REFERENCE)
    return [
        timing.wall_time(
            call,
            label=f"dtw_batch / {device.type}",
            sub_label=cases.batch_label(config, spread),
            description=name,
            min_run_time=min_run_time,
            num_threads=torch.get_num_threads(),
        )
        for name, call in calls.items()
    ]


def benchmark(min_run_time: float = 0.2, output: Path | None = None) -> None:
    """Benchmark DTW, single-pair then batched. With ``output``, inject the results into a markdown file."""
    devices = cases.devices()
    single = [
        m
        for device in devices
        for shape in cases.SHAPES
        for m in single_measurements(shape, device, min_run_time=min_run_time)
    ]
    batched = [
        m
        for device in devices
        for config in cases.BATCH_CONFIGS
        for spread in cases.SPREADS
        for m in batch_measurements(config, spread, device, min_run_time=min_run_time)
    ]
    report.show(single)
    report.show(batched)
    if output is not None:
        report.inject(output, single + batched, MARKERS)
