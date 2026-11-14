"""DTW benchmark: compare torchdtw's ``dtw`` and ``dtw_batch`` against the other DTW implementations.

Each batch config runs at every spread of sequence lengths of ``cases.SPREADS``, with int32 lengths. torchdtw is
the correctness reference of every case. Each implementation runs in its own process (``timing.each_in_own_process``).
Tables have one row per case and one column per implementation.
"""

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.benchmark import Measurement

from dtw_benchmark import cases, report, timing

from .implementations import NAIVE, REFERENCE, available

MARKERS = ("<!-- benchmark -->", "<!-- /benchmark -->")

BatchCase = tuple[tuple[int, int, int, int, bool], str]


def single_worker(
    name: str,
    shapes: Sequence[tuple[int, int]],
    references: Mapping[str, torch.Tensor] | None,
    *,
    device: torch.device,
    min_run_time: float,
) -> tuple[dict[str, torch.Tensor], list[Measurement]]:
    """Time the single-pair implementation ``name`` on each shape, after checking it against torchdtw."""
    fn = available(device, batched=False)[name]
    outputs, measurements = {}, []
    for shape in shapes:
        sub_label = cases.shape_label(shape)
        call = functools.partial(fn, cases.single_input(shape, device=device))
        outputs[sub_label] = timing.warm_up(call).cpu()
        if references is not None:
            timing.check(name, outputs[sub_label], references[sub_label])
        measurements.append(
            timing.wall_time(
                call,
                label=f"dtw / {device.type}",
                sub_label=sub_label,
                description=name,
                min_run_time=min_run_time,
                num_threads=torch.get_num_threads(),
            )
        )
    return outputs, measurements


def batch_worker(
    name: str,
    batch_cases: Sequence[BatchCase],
    references: Mapping[str, torch.Tensor] | None,
    *,
    device: torch.device,
    min_run_time: float,
) -> tuple[dict[str, torch.Tensor], list[Measurement]]:
    """Time the batched implementation ``name`` on each config and spread, after checking it against torchdtw."""
    fn = available(device, batched=True)[name]
    outputs, measurements = {}, []
    for config, spread in batch_cases:
        sub_label = cases.batch_label(config, spread)
        d, sx, sy = cases.batch_inputs(config, spread, device=device)
        call = functools.partial(fn, d, sx, sy, symmetric=config[-1])
        outputs[sub_label] = timing.warm_up(call).cpu()
        if references is not None:
            timing.check(name, outputs[sub_label], references[sub_label])
        measurements.append(
            timing.wall_time(
                call,
                label=f"dtw_batch / {device.type}",
                sub_label=sub_label,
                description=name,
                min_run_time=min_run_time,
                num_threads=torch.get_num_threads(),
            )
        )
    return outputs, measurements


def benchmark(min_run_time: float = 0.2, output: Path | None = None) -> None:
    """Benchmark DTW, single-pair then batched. With ``output``, inject the results into a markdown file."""
    single, batched = [], []
    batch_cases = [(config, spread) for config in cases.BATCH_CONFIGS for spread in cases.SPREADS]
    for device in cases.devices():
        shapes = {
            # O(N*M) Python loop: only on the smallest shape
            name: cases.SHAPES[:1] if name == NAIVE else cases.SHAPES
            for name in available(device, batched=False)
        }
        single += timing.each_in_own_process(
            single_worker, shapes, REFERENCE, device=device, min_run_time=min_run_time
        )
        batched += timing.each_in_own_process(
            batch_worker,
            dict.fromkeys(available(device, batched=True), batch_cases),
            REFERENCE,
            device=device,
            min_run_time=min_run_time,
        )
    report.show(single)
    report.show(batched)
    if output is not None:
        report.inject(output, single + batched, MARKERS)
