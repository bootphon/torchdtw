"""Soft-DTW benchmark: compare torchdtw's ``soft_dtw`` against the other implementations, forward and backward.

Each paired config runs at every spread of sequence lengths of ``cases.SPREADS``, with int32 lengths. torchdtw is
the correctness reference of every case, for the costs and for the gradients with respect to the sequences. Each
implementation runs in its own process (``timing.each_in_own_process``). Tables have one row per case and one column
per implementation.
"""

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.benchmark import Measurement

from dtw_benchmark import cases, report, soft_cases, timing

from .implementations import REFERENCE, available

MARKERS = ("<!-- benchmark-soft -->", "<!-- /benchmark-soft -->")
ATOL = RTOL = 1e-3
GRAD_TOLERANCE = 1e-2

SoftCase = tuple[tuple[int, int, int], str]
# Costs, and the gradient with respect to x when timing the backward pass.
Outputs = tuple[torch.Tensor, torch.Tensor | None]


def worker(
    name: str,
    configs: Sequence[SoftCase],
    references: Mapping[str, Outputs] | None,
    *,
    device: torch.device,
    backward: bool,
    min_run_time: float,
) -> tuple[dict[str, Outputs], list[Measurement]]:
    """Time the implementation ``name`` on each config and spread, after checking it against torchdtw.

    With ``backward``, each call is a forward and a backward pass, to the gradient with respect to ``x``.
    """
    fn = available(device)[name]
    outputs, measurements = {}, []
    for config, spread in configs:
        sub_label = soft_cases.config_label(config, spread)
        x, y, sx, sy = soft_cases.sequences(config, spread, device=device)
        costs = functools.partial(fn, y=y, sx=sx, sy=sy, gamma=soft_cases.GAMMA)
        call = functools.partial(soft_cases.gradient, costs, x) if backward else functools.partial(costs, x)
        cost = timing.warm_up(functools.partial(costs, x)).cpu()
        grad = timing.warm_up(call).cpu() if backward else None
        outputs[sub_label] = (cost, grad)
        if references is not None:
            expected_cost, expected_grad = references[sub_label]
            timing.check(name, cost, expected_cost, atol=ATOL, rtol=RTOL)
            if grad is not None:
                scale = expected_grad.abs().max().item()
                timing.check(name, grad, expected_grad, atol=GRAD_TOLERANCE * scale, rtol=GRAD_TOLERANCE)
        measurements.append(
            timing.wall_time(
                call,
                label=f"soft_dtw / {device.type} ({'forward + backward' if backward else 'forward'})",
                sub_label=sub_label,
                description=name,
                min_run_time=min_run_time,
                num_threads=torch.get_num_threads(),
            )
        )
    return outputs, measurements


def benchmark(min_run_time: float = 0.2, output: Path | None = None, *, backward: bool = True) -> None:
    """Benchmark soft-DTW. With ``output``, inject the results into a markdown file."""
    all_cases = [(config, spread) for config in soft_cases.CONFIGS for spread in cases.SPREADS]
    results = [
        m
        for device in cases.devices()
        for m in timing.each_in_own_process(
            worker,
            dict.fromkeys(available(device), all_cases),
            REFERENCE,
            device=device,
            backward=backward,
            min_run_time=min_run_time,
        )
    ]
    report.show(results)
    if output is not None:
        report.inject(output, results, MARKERS)
