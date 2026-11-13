"""Command-line interface for the benchmarks.

Two subcommands, each dispatching to its own logic module:
- ``run dtw``: benchmark torchdtw against the other DTW implementations (``dtw.run_benchmark``).
- ``compare``: compare two torchdtw builds, by default the latest release against the installed one
  (``compare_release``).

The modules are imported lazily so that running one subcommand does not pay for the other's imports.
"""

import argparse
from pathlib import Path


def _add_run_options(parser: argparse.ArgumentParser, markers: str) -> None:
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        metavar="SECONDS",
        help="Minimum measurement time per case passed to torch's blocked_autorange (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help=f"Markdown file to inject the results tables into, between the {markers} markers.",
    )


def main() -> None:
    """Parse the command line and dispatch to the selected subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m dtw_benchmark",
        description=(
            "Benchmarking tools for torchdtw.\n\n"
            "Each subcommand runs on the CPU and, when available, on CUDA. Every run checks correctness before timing."
        ),
        epilog=(
            "examples:\n"
            "  python -m dtw_benchmark run dtw\n"
            "  python -m dtw_benchmark run dtw --output README.md\n"
            "  python -m dtw_benchmark compare\n"
            "  python -m dtw_benchmark compare dist/base/torchdtw-*.whl dist/new/torchdtw-*.whl --timer gpu\n"
            "  python -m dtw_benchmark compare torchdtw==1.2.3 installed --only 'cuda n=16x16 s=32x32'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{run,compare}")

    run_parser = sub.add_parser(
        "run",
        help="Benchmark torchdtw against other implementations",
        description="Benchmark torchdtw against other implementations and print timing comparison tables.",
    )
    run_sub = run_parser.add_subparsers(dest="target", required=True, metavar="{dtw}")
    dtw_parser = run_sub.add_parser(
        "dtw",
        help="dtw and dtw_batch against Cython, Numba, Triton and a naive PyTorch loop",
        description=(
            "Benchmark torchdtw's dtw then dtw_batch against the other DTW implementations "
            "(Cython, Numba, Triton, and a naive PyTorch loop)."
        ),
    )
    _add_run_options(dtw_parser, "<!-- benchmark -->")

    compare_parser = sub.add_parser(
        "compare",
        help="Compare two torchdtw builds",
        description=(
            "Compare two torchdtw builds for both correctness and speed, on identical inputs. BASELINE and "
            "CANDIDATE are each a wheel path, a version spec (torchdtw==1.2.3), or 'installed': the torchdtw of the "
            "current environment. Each side other than 'installed' is installed without dependencies into a "
            "temporary directory, then each side is measured in a child process of this interpreter, so that both "
            "share the same torch and only torchdtw differs."
        ),
    )
    compare_parser.add_argument(
        "baseline",
        nargs="?",
        metavar="BASELINE",
        help="Reference build (default: the latest torchdtw release on PyPI).",
    )
    compare_parser.add_argument(
        "candidate",
        nargs="?",
        default="installed",
        metavar="CANDIDATE",
        help="Build to compare against the baseline (default: %(default)s).",
    )
    compare_parser.add_argument(
        "--timer",
        choices=["wall", "gpu"],
        default="wall",
        help=(
            "wall: wall-clock time per call, with a CUDA sync after each call, so it includes the host overhead. "
            "gpu: GPU time per call, from CUDA events around a loop of calls; CPU cases stay wall-clock "
            "(default: %(default)s)."
        ),
    )
    compare_parser.add_argument(
        "--only",
        metavar="SUBSTRING",
        help="Only measure the cases whose label (e.g. 'dtw_batch / cuda n=16x16 s=32x32 sym none int32') "
        "contains SUBSTRING.",
    )
    compare_parser.add_argument(
        "--min-run-time",
        type=float,
        default=1,
        metavar="SECONDS",
        help="Minimum measurement time per case (default: %(default)s).",
    )
    compare_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random inputs, shared by both builds (default: %(default)s).",
    )

    args = parser.parse_args()
    if args.command == "run":
        from .dtw.run_benchmark import benchmark

        benchmark(args.min_run_time, args.output)
    else:
        from .compare_release import driver

        driver(
            args.baseline,
            args.candidate,
            timer=args.timer,
            only=args.only,
            min_run_time=args.min_run_time,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
