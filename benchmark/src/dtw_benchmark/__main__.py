import torch
from torch.utils.benchmark import Compare, Timer


def get_timers(n: int, label: str = "CPU"):
    x = torch.randn((n, n))
    t_torch = Timer(
        stmt="dtw_torch(x)",
        setup="from dtw_benchmark.impl import dtw_torch",
        globals={"x": x},
        num_threads=num_threads,
        label=label,
        sub_label="PyTorch naive",
        description=str(n),
    )
    t_cython = Timer(
        stmt="dtw_cython(x)",
        setup="from dtw_benchmark.impl import dtw_cython",
        globals={"x": x},
        num_threads=num_threads,
        label="CPU",
        sub_label="Cython",
        description=str(n),
    )
    t_numba = Timer(
        stmt="dtw_numba(x)",
        setup="from dtw_benchmark.impl import dtw_numba",
        globals={"x": x},
        num_threads=num_threads,
        label=label,
        sub_label="Numba",
        description=str(n),
    )
    t_torchdtw = Timer(
        stmt="dtw(x)",
        setup="from torchdtw import dtw",
        globals={"x": x},
        num_threads=num_threads,
        label=label,
        sub_label="PyTorch C++ extension",
        description=str(n),
    )
    return [
        t_torch.blocked_autorange(min_run_time=min_run_time),
        t_cython.blocked_autorange(min_run_time=min_run_time),
        t_numba.blocked_autorange(min_run_time=min_run_time),
        t_torchdtw.blocked_autorange(min_run_time=min_run_time),
    ]


if __name__ == "__main__":
    num_threads = torch.get_num_threads()
    min_run_time = 0.2
    results = []
    for n in [16, 32, 64, 128, 256, 512]:
        results.extend(get_timers(n))
    compare = Compare(results)
    compare.colorize()
    compare.print()
