"""Command-line interface for the DTW benchmark.

Two subcommands, each dispatching to its own logic module:
- ``run``: benchmark torchdtw against the other implementations (``run_benchmark``).
- ``compare``: compare the current checkout against the latest released torchdtw (``compare_release``).

Both modules are imported lazily so that running one subcommand does not pay for the other's imports.
"""

import argparse
from pathlib import Path


def main() -> None:
    """Parse the command line and dispatch to the selected subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m dtw_benchmark",
        description=(
            "Benchmarking tools for torchdtw's dtw and dtw_batch functions.\n\n"
            "Each subcommand runs on the CPU and, when available, on CUDA, sweeping a range of "
            "single-pair shapes and batched configurations. Every run checks correctness before timing."
        ),
        epilog=(
            "examples:\n"
            "  python -m dtw_benchmark run\n"
            "  python -m dtw_benchmark run --output README.md\n"
            "  python -m dtw_benchmark compare\n"
            "  python -m dtw_benchmark compare --wheel dist/torchdtw-1.2.3-*.whl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{run,compare}")

    run_parser = sub.add_parser(
        "run",
        help="Benchmark torchdtw against other DTW implementations",
        description=(
            "Benchmark torchdtw's C++/CUDA extension against the other DTW implementations "
            "(Cython, Numba, Triton, and a naive PyTorch loop) and print a timing comparison table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        metavar="SECONDS",
        help="Minimum measurement time per case passed to torch's blocked_autorange (default: %(default)s).",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Markdown file to inject the results table into, between the <!-- benchmark --> markers.",
    )

    compare_parser = sub.add_parser(
        "compare",
        help="Compare the current checkout against the latest released torchdtw",
        description=(
            "Compare the current torchdtw checkout against a released build for both correctness and "
            "speed. The reference is measured in an isolated 'uv run' environment on identical inputs; "
            "by default it is the latest release on PyPI, or a local wheel given with --wheel."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare_parser.add_argument(
        "--min-run-time",
        type=float,
        default=1,
        metavar="SECONDS",
        help="Minimum measurement time per case passed to torch's blocked_autorange (default: %(default)s).",
    )
    compare_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random input tensors, shared by both versions (default: %(default)s).",
    )
    compare_parser.add_argument(
        "--wheel",
        type=Path,
        metavar="WHEEL",
        help="Reference torchdtw wheel to compare against, instead of the latest release on PyPI.",
    )

    args = parser.parse_args()
    if args.command == "run":
        from .run_benchmark import benchmark  # noqa: PLC0415

        benchmark(args.min_run_time, args.output)
    else:
        from .compare_release import driver  # noqa: PLC0415

        driver(min_run_time=args.min_run_time, seed=args.seed, path_wheel=args.wheel)


if __name__ == "__main__":
    main()
