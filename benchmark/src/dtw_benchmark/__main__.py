"""Run the benchmark."""

import torch
from torch.utils.benchmark import Compare, Measurement, Timer

from . import dtw, dtw_cython, dtw_numba, dtw_torch, dtw_triton

DIMENSIONS = [16, 32, 64, 128, 256, 512]


def measurements(dim: int, device: torch.device, min_run_time: float = 0.2) -> list[Measurement]:
    """Measure DTW execution time."""
    num_threads = torch.get_num_threads()
    x = torch.testing.make_tensor((dim, dim), dtype=torch.float32, device=device)
    outputs = [d(x) for d in [dtw, dtw_cython, dtw_numba, dtw_torch] + ([dtw_triton] if x.is_cuda else [])]
    for out in outputs[1:]:
        torch.testing.assert_close(out, outputs[0])

    def measure(function: str, sub_label: str) -> Measurement:
        return Timer(
            stmt=f"{function}(x)",
            setup=f"from dtw_benchmark import {function}",
            globals={"x": x},
            num_threads=num_threads,
            label=device.type,
            sub_label=sub_label,
            description=str(dim),
        ).blocked_autorange(min_run_time=min_run_time)

    return ([measure("dtw_torch", "PyTorch naive")] if dim == DIMENSIONS[0] else []) + (
        [measure("dtw_cython", "Cython"), measure("dtw_numba", "Numba")]
        + ([measure("dtw_triton", "Triton")] if x.is_cuda else [])
        + [measure("dtw", "PyTorch C++ extension")]
    )


def benchmark(min_run_time: float = 0.2) -> None:
    """Benchmark DTW."""
    results = []
    for device_type in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for dim in DIMENSIONS:
            results.extend(measurements(dim, torch.device(device_type), min_run_time))
    compare = Compare(results)
    compare.colorize()
    compare.print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-run-time", type=float, default=0.2)
    args = parser.parse_args()
    benchmark(args.min_run_time)
