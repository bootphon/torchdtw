"""Timing helpers shared by ``run`` and ``compare``: wall-clock and GPU time, and the warm-up + correctness check.

Torch-only, so that ``compare_release.py``, run as a script, can import it as a sibling module.
"""

from collections.abc import Callable, Mapping

import torch
from torch.utils.benchmark import Measurement, Timer
from torch.utils.benchmark.utils.common import TaskSpec

# Smallest duration of one block of calls timed by ``gpu_time``, so that the event resolution does not matter.
GPU_BLOCK_TIME = 1e-2


def wall_time(
    fn: Callable[[], object],
    *,
    label: str,
    sub_label: str,
    description: str,
    min_run_time: float,
    num_threads: int = 1,
    sync_each_call: bool = False,
) -> Measurement:
    """Wall-clock time per call of ``fn``, with ``Timer.blocked_autorange``.

    ``Timer`` synchronizes CUDA once per block of calls: that is the throughput of back-to-back calls. With
    ``sync_each_call`` (CUDA cases only), CUDA is synchronized after every call instead, so the time includes the
    host overhead of each call: the latency.
    """
    if sync_each_call:

        def call() -> None:
            fn()
            torch.cuda.synchronize()

    else:
        call = fn
    return Timer(
        stmt="call()",
        globals={"call": call},
        num_threads=num_threads,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def gpu_time(
    fn: Callable[[], object],
    *,
    label: str,
    sub_label: str,
    description: str,
    min_run_time: float,
) -> Measurement:
    """GPU time per call of ``fn``, from CUDA events around blocks of back-to-back calls.

    The number of calls per block doubles until a block takes ``GPU_BLOCK_TIME``, then blocks run until
    ``min_run_time`` is spent.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def block(number: int) -> float:
        start.record()
        for _ in range(number):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 1e3

    number = 1
    while (elapsed := block(number)) < GPU_BLOCK_TIME:
        number *= 2
    raw_times = [elapsed]
    while sum(raw_times) < min_run_time:
        raw_times.append(block(number))
    task_spec = TaskSpec(stmt="fn()", setup="pass", label=label, sub_label=sub_label, description=description)
    return Measurement(number_per_run=number, raw_times=raw_times, task_spec=task_spec)


def warm_up_and_check(
    calls: Mapping[str, Callable[[], torch.Tensor]],
    reference: str,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict[str, torch.Tensor]:
    """Call each implementation twice, then check its output against the ``reference`` one.

    The warm-up triggers JIT compilation (Numba, Triton) and fills the allocator caches. ``atol`` and ``rtol`` are
    the tolerances of ``torch.testing.assert_close``, per runner (both or neither). Returns the outputs, by name.
    """
    outputs = {}
    for name, call in calls.items():
        for _ in range(2):
            outputs[name] = call()
    for name, out in outputs.items():
        torch.testing.assert_close(
            out, outputs[reference], atol=atol, rtol=rtol, msg=lambda m, name=name: f"{name}: {m}"
        )
    return outputs
