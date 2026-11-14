"""Soft-DTW benchmark: compare torchdtw's ``soft_dtw`` against the other implementations, forward and backward.

Each paired config runs at every spread of sequence lengths of ``cases.SPREADS``, with int32 lengths. torchdtw is
the correctness reference of every case, for the costs and for the gradients with respect to the sequences. Tables
have one row per case and one column per implementation.
"""

import functools
from pathlib import Path

import torch
from torch.utils.benchmark import Measurement

from dtw_benchmark import cases, report, soft_cases, timing

from .implementations import REFERENCE, available

MARKERS = ("<!-- benchmark-soft -->", "<!-- /benchmark-soft -->")
ATOL = RTOL = 1e-3
GRAD_TOLERANCE = 1e-2


def measurements(
    config: tuple[int, int, int], spread: str, device: torch.device, *, backward: bool, min_run_time: float
) -> list[Measurement]:
    """Time every implementation on one config and spread, after checking it against torchdtw.

    With ``backward``, each call is a forward and a backward pass, to the gradient with respect to ``x``.
    """
    x, y, sx, sy = soft_cases.sequences(config, spread, device=device)
    impls = available(device)
    costs = {name: functools.partial(fn, y=y, sx=sx, sy=sy, gamma=soft_cases.GAMMA) for name, fn in impls.items()}
    timing.warm_up_and_check(
        {name: functools.partial(fn, x) for name, fn in costs.items()}, REFERENCE, atol=ATOL, rtol=RTOL
    )
    if backward:
        calls = {name: functools.partial(soft_cases.gradient, fn, x) for name, fn in costs.items()}
        scale = calls[REFERENCE]().abs().max().item()
        timing.warm_up_and_check(calls, REFERENCE, atol=GRAD_TOLERANCE * scale, rtol=GRAD_TOLERANCE)
    else:
        calls = {name: functools.partial(fn, x) for name, fn in costs.items()}
    return [
        timing.wall_time(
            call,
            label=f"soft_dtw / {device.type} ({'forward + backward' if backward else 'forward'})",
            sub_label=soft_cases.config_label(config, spread),
            description=name,
            min_run_time=min_run_time,
            num_threads=torch.get_num_threads(),
        )
        for name, call in calls.items()
    ]


def benchmark(min_run_time: float = 0.2, output: Path | None = None, *, backward: bool = True) -> None:
    """Benchmark soft-DTW. With ``output``, inject the results into a markdown file."""
    results = [
        m
        for device in cases.devices()
        for config in soft_cases.CONFIGS
        for spread in cases.SPREADS
        for m in measurements(config, spread, device, backward=backward, min_run_time=min_run_time)
    ]
    report.show(results)
    if output is not None:
        report.inject(output, results, MARKERS)
