"""Standard benchmark: compare torchdtw against the other DTW implementations."""

import contextlib
import io
import platform
import re
from pathlib import Path

import torch
from torch.utils.benchmark import Compare, Measurement, Timer

from . import (
    dtw,
    dtw_batch,
    dtw_cython,
    dtw_cython_batch,
    dtw_numba,
    dtw_numba_batch,
    dtw_torch,
    dtw_triton,
    dtw_triton_batch,
)

# Square shapes plus a couple of rectangular (N != M) ones to exercise the off-diagonal boundaries.
SHAPES = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (128, 512), (512, 128)]
# Batched configs: (n1, n2, s1, s2, symmetric) -- spanning few-large-pairs to the many-small-pairs
# workload torchdtw targets.
BATCH_CONFIGS = [
    (16, 16, 128, 128, True),
    (32, 32, 64, 64, False),
    (64, 64, 32, 32, True),
    (128, 128, 16, 16, True),
    (128, 128, 16, 16, False),
    (256, 256, 8, 8, True),
]
SEED = 0

# Markers in the --output markdown file between which results are injected, as in the repo's docs.py.
BENCHMARK_MARKERS = ("<!-- benchmark -->", "<!-- /benchmark -->")

FUNCS = {
    "dtw": dtw,
    "dtw_cython": dtw_cython,
    "dtw_numba": dtw_numba,
    "dtw_torch": dtw_torch,
    "dtw_triton": dtw_triton,
}
BATCH_FUNCS = {
    "dtw_batch": dtw_batch,
    "dtw_cython_batch": dtw_cython_batch,
    "dtw_numba_batch": dtw_numba_batch,
    "dtw_triton_batch": dtw_triton_batch,
}


def measurements(shape: tuple[int, int], device: torch.device, min_run_time: float = 0.2) -> list[Measurement]:
    """Measure DTW execution time. The C++ extension is used as the correctness reference."""
    num_threads = torch.get_num_threads()
    g = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.rand(shape, generator=g, dtype=torch.float32).to(device)

    # Display order; the naive O(N*M) Python loop is only run on the smallest shape.
    impls: list[tuple[str, str]] = []
    if shape == SHAPES[0]:
        impls.append(("PyTorch naive", "dtw_torch"))
    impls += [("Cython", "dtw_cython"), ("Numba", "dtw_numba")]
    if x.is_cuda:
        impls.append(("Triton", "dtw_triton"))
    impls.append(("PyTorch C++ extension", "dtw"))

    # Warm up (triggers Numba/Triton JIT compilation and allocator caching), then check correctness.
    outputs = {}
    for _, name in impls:
        for _ in range(2):
            out = FUNCS[name](x)
        outputs[name] = out
    for name, out in outputs.items():
        if name != "dtw":
            torch.testing.assert_close(out, outputs["dtw"])

    description = f"{shape[0]}x{shape[1]}"

    def measure(function: str, sub_label: str) -> Measurement:
        return Timer(
            stmt=f"{function}(x)",
            setup=f"from dtw_benchmark import {function}",
            globals={"x": x},
            num_threads=num_threads,
            label=device.type,
            sub_label=sub_label,
            description=description,
        ).blocked_autorange(min_run_time=min_run_time)

    return [measure(name, sub_label) for sub_label, name in impls]


def batch_measurements(
    config: tuple[int, int, int, int, bool], device: torch.device, min_run_time: float = 0.2
) -> list[Measurement]:
    """Measure batched DTW execution time. The C++ extension is used as the correctness reference."""
    num_threads = torch.get_num_threads()
    n1, n2, s1, s2, symmetric = config
    g = torch.Generator(device="cpu").manual_seed(SEED)
    d = torch.rand((n1, n2, s1, s2), generator=g, dtype=torch.float32)
    if symmetric:
        n = min(n1, n2)
        i, j = torch.triu_indices(n, n)
        d[:n, :n][i, j] = d[:n, :n][j, i]
    sx = torch.randint(1, s1 + 1, (n1,), generator=g, dtype=torch.long)
    sy = sx.clone() if symmetric and n1 == n2 else torch.randint(1, s2 + 1, (n2,), generator=g, dtype=torch.long)
    d, sx, sy = d.to(device), sx.to(device), sy.to(device)

    impls = [("Cython", "dtw_cython_batch"), ("Numba", "dtw_numba_batch")]
    if d.is_cuda:
        impls.append(("Triton", "dtw_triton_batch"))
    impls.append(("PyTorch C++ extension", "dtw_batch"))

    # Warm up (triggers Numba/Triton JIT compilation and allocator caching), then check correctness.
    outputs = {}
    for _, name in impls:
        for _ in range(2):
            out = BATCH_FUNCS[name](d, sx, sy, symmetric=symmetric)
        outputs[name] = out
    for name, out in outputs.items():
        if name != "dtw_batch":
            torch.testing.assert_close(out, outputs["dtw_batch"])

    description = f"n={n1}x{n2} s={s1}x{s2} {'sym' if symmetric else 'asym'}"

    def measure(function: str, sub_label: str) -> Measurement:
        return Timer(
            stmt=f"{function}(d, sx, sy, symmetric=sym)",
            setup=f"from dtw_benchmark import {function}",
            globals={"d": d, "sx": sx, "sy": sy, "sym": symmetric},
            num_threads=num_threads,
            label=f"{device.type} (batch)",
            sub_label=sub_label,
            description=description,
        ).blocked_autorange(min_run_time=min_run_time)

    return [measure(name, sub_label) for sub_label, name in impls]


def _device_name() -> str:
    """Name of the accelerator the benchmark runs on, falling back to the CPU."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine() or "CPU"


def _render(results: list[Measurement]) -> str:
    """Render measurements as the plain (uncolored) table text that ``Compare`` prints."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        Compare(results).print()
    return buffer.getvalue().strip("\n")


def _inject(output: Path, results: list[Measurement]) -> None:
    """Inject the rendered results between the benchmark markers of the ``output`` markdown file."""
    begin, end = BENCHMARK_MARKERS
    block = f"## Benchmark results on {_device_name()}\n\n```\n{_render(results)}\n```"
    content = output.read_text(encoding="utf-8")
    pattern = re.escape(begin) + r".*?" + re.escape(end)
    new_content, count = re.subn(pattern, lambda _: f"{begin}\n{block}\n{end}", content, flags=re.DOTALL)
    if count == 0:
        msg = f"Markers not found in {output}. Add:\n{begin}\n{end}"
        raise RuntimeError(msg)
    output.write_text(new_content, encoding="utf-8")


def benchmark(min_run_time: float = 0.2, output: Path | None = None) -> None:
    """Benchmark DTW, single-pair then batched. With ``output``, inject the results into a markdown file."""
    devices = [torch.device(t) for t in ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])]

    single = [m for device in devices for shape in SHAPES for m in measurements(shape, device, min_run_time)]
    batched = [m for device in devices for cfg in BATCH_CONFIGS for m in batch_measurements(cfg, device, min_run_time)]
    for results in (single, batched):
        compare = Compare(results)
        compare.colorize()
        compare.print()
    if output is not None:
        _inject(output, single + batched)
