import torch
from torch.utils.benchmark import Compare, Measurement, Timer

from dtw_benchmark import dtw, dtw_cython, dtw_numba, dtw_torch, dtw_triton


def get_measurements(
    n: int,
    device: torch.device,
    min_run_time: float = 0.2,
) -> list[Measurement]:
    num_threads = torch.get_num_threads()
    x = torch.testing.make_tensor((n, n), dtype=torch.float32, device=device)

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
            description=str(n),
        ).blocked_autorange(min_run_time=min_run_time)

    return ([measure("dtw_torch", "PyTorch naive")] if n < 20 else []) + (
        [measure("dtw_cython", "Cython"), measure("dtw_numba", "Numba")]
        + ([measure("dtw_triton", "Triton")] if x.is_cuda else [])
        + [measure("dtw", "PyTorch C++ extension")]
    )


def benchmark(min_run_time: float = 0.2) -> None:
    dims, device_types = [16, 32, 64, 128, 256, 512, 1023], ["cpu", "cuda"]
    results = []
    for device_type in device_types:
        for n in dims:
            results.extend(get_measurements(n, torch.device(device_type), min_run_time))
    compare = Compare(results)
    compare.colorize()
    compare.print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-run-time", type=float, default=0.2)
    args = parser.parse_args()
    benchmark(args.min_run_time)
