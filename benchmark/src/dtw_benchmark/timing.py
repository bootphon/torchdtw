"""Timing helpers shared by ``run`` and ``compare``: wall-clock and GPU time, warm-up, checks, process isolation.

Torch-only, so that ``compare_release.py``, run as a script, can import it as a sibling module.
"""

import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch
from torch.utils.benchmark import Measurement, Timer
from torch.utils.benchmark.utils.common import TaskSpec

# Smallest duration of one block of calls timed by ``gpu_time``, so that the event resolution does not matter.
GPU_BLOCK_TIME = 1e-2

# ``worker(name, cases, references, **kwargs) -> (outputs by case label, measurements)``: see ``each_in_own_process``.
Worker = Callable[..., tuple[dict[str, Any], list[Measurement]]]


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


def warm_up(call: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Call ``call`` twice and return its last output.

    The warm-up triggers JIT compilation (Numba, Triton) and fills the allocator caches.
    """
    call()
    return call()


def check(
    name: str, out: torch.Tensor, expected: torch.Tensor, *, atol: float | None = None, rtol: float | None = None
) -> None:
    """Check the output of implementation ``name`` against the reference one, with ``torch.testing.assert_close``.

    ``atol`` and ``rtol`` are its tolerances (both or neither).
    """
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol, msg=lambda m: f"{name}: {m}")


def in_own_process[T](fn: Callable[..., T], /, *args: object, **kwargs: object) -> T:
    """Return ``fn(*args, **kwargs)``, run in a new process, started with ``spawn``."""
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(fn, *args, **kwargs).result()


def each_in_own_process(
    worker: Worker, cases_by_name: Mapping[str, Sequence[object]], reference: str, **kwargs: object
) -> list[Measurement]:
    """Time each implementation in its own process, the reference first, and return the measurements in order.

    ``worker(name, cases, references, **kwargs)`` warms up and times implementation ``name`` on each of its
    ``cases``, and returns its outputs, by case label and on the CPU, with its measurements. It checks each output
    against ``references``, the outputs of ``reference``, before timing it. ``references`` is None for ``reference``
    itself.

    A process per implementation keeps the thread pools of one implementation from slowing down another: Numba's
    ``omp`` threading layer loads a second OpenMP runtime next to PyTorch's, and once its threads exist the small
    matrix products of ``torch.cdist`` ran ~60x slower on a shared 24-core allocation. The measurements are in the
    order of ``cases_by_name``, then of each implementation's cases, which sets the column and row order of the tables.
    """
    references, by_name = in_own_process(worker, reference, cases_by_name[reference], None, **kwargs)
    measurements = {reference: by_name}
    for name, name_cases in cases_by_name.items():
        if name != reference:
            measurements[name] = in_own_process(worker, name, name_cases, references, **kwargs)[1]
    return [m for name in cases_by_name for m in measurements[name]]
