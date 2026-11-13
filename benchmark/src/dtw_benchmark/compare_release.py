"""Compare torchdtw's ``dtw`` and ``dtw_batch`` between two builds: correctness, then speed.

Driven by the ``compare`` subcommand of ``python -m dtw_benchmark`` (the CLI lives in ``__main__``). Each side, the
baseline and the candidate, is a wheel path, a version spec (``torchdtw==1.2.3``) or ``installed``, the torchdtw of
the current environment. ``driver`` installs each side other than ``installed`` into its own temporary directory
with ``uv pip install --no-deps --target``, then measures both sides the same way, each in a child process: this
interpreter runs this module as a script, with the side's directory first on ``PYTHONPATH`` so that its torchdtw
shadows the installed one. Both sides then share the interpreter and torch, and only torchdtw differs.

Running this module as a script puts its own directory on ``sys.path``: the child imports ``cases`` and ``timing``
as sibling modules, which only depend on torch.
"""

import argparse
import json
import os
import pickle
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch.utils.benchmark import Compare, Measurement

if __package__:
    from . import cases, timing
else:  # run as a script by ``driver``
    import cases  # ty: ignore[import-not-found]
    import timing  # ty: ignore[import-not-found]

INSTALLED = "installed"
# Changes of the median beyond this fraction are listed after the tables.
THRESHOLD = 0.01


def latest_version() -> str:
    """Query PyPI to get the latest torchdtw version."""
    cmd = ["uvx", "pip", "index", "versions", "torchdtw", "--json"]
    process = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(process.stdout)["latest"]


# --- Child: measure the torchdtw it imports ---


def _time(
    call: Callable[[], object],
    device: torch.device,
    *,
    timer: str,
    num_threads: int,
    label: str,
    sub_label: str,
    description: str,
    min_run_time: float,
) -> Measurement:
    """GPU time of a CUDA case with ``--timer gpu``, else wall-clock time, with a CUDA sync after each call."""
    if timer == "gpu" and device.type == "cuda":
        return timing.gpu_time(
            call, label=label, sub_label=sub_label, description=description, min_run_time=min_run_time
        )
    return timing.wall_time(
        call,
        label=label,
        sub_label=sub_label,
        description=description,
        min_run_time=min_run_time,
        num_threads=num_threads,
        sync_each_call=device.type == "cuda",
    )


def measure_all(
    description: str, *, timer: str, only: str | None, min_run_time: float, seed: int
) -> tuple[dict, list[Measurement]]:
    """Measure ``dtw`` and ``dtw_batch`` on every case whose ``"<label> <sub_label>"`` contains ``only``.

    Batch cases run with int32 and int64 lengths, and on the CPU with 1 thread and with all of them.
    """
    import torchdtw  # only the child imports torchdtw

    outputs, timings = {}, []

    def measure(
        fn: Callable[[], torch.Tensor], device: torch.device, label: str, sub_label: str, num_threads: int = 1
    ) -> None:
        if only is not None and only not in f"{label} {sub_label}":
            return
        fn()  # warm-up
        outputs[label, sub_label] = fn().detach().cpu().clone()
        timings.append(
            _time(
                fn,
                device,
                timer=timer,
                num_threads=num_threads,
                label=label,
                sub_label=sub_label,
                description=description,
                min_run_time=min_run_time,
            )
        )

    for device in cases.devices():
        for shape in cases.SHAPES:
            x = cases.single_input(shape, device=device, seed=seed)
            measure(lambda x=x: torchdtw.dtw(x), device, f"dtw / {device.type}", cases.shape_label(shape))
        thread_counts = sorted({1, torch.get_num_threads()}) if device.type == "cpu" else [1]
        for config in cases.BATCH_CONFIGS:
            for spread in cases.SPREADS:
                for dtype in (torch.int32, torch.int64):
                    d, sx, sy = cases.batch_inputs(config, spread, device=device, seed=seed, lengths_dtype=dtype)
                    for num_threads in thread_counts:
                        threads = f" t={num_threads}" if device.type == "cpu" else ""
                        measure(
                            lambda d=d, sx=sx, sy=sy, sym=config[-1]: torchdtw.dtw_batch(d, sx, sy, symmetric=sym),
                            device,
                            f"dtw_batch / {device.type}",
                            f"{cases.batch_label(config, spread)} {str(dtype).removeprefix('torch.')}{threads}",
                            num_threads,
                        )
    return outputs, timings


def _installed_version() -> str:
    try:
        return version("torchdtw")
    except PackageNotFoundError:
        return "unknown"


def _child_main() -> None:
    """Child entry point: measure the torchdtw it imports, and dump the results with pickle."""
    parser = argparse.ArgumentParser(description="Measure one torchdtw build (internal child of 'compare').")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the pickled measurements to.")
    parser.add_argument("--side", required=True, help="Side of the comparison: 'baseline' or 'candidate'.")
    parser.add_argument("--min-run-time", type=float, required=True, help="Minimum measurement time per case.")
    parser.add_argument("--seed", type=int, required=True, help="Seed for the random inputs.")
    parser.add_argument("--timer", choices=["wall", "gpu"], required=True, help="How to time the CUDA cases.")
    parser.add_argument("--only", help="Only measure the cases whose label contains this substring.")
    parser.add_argument("--expect-dir", type=Path, help="Directory that the imported torchdtw must come from.")
    args = parser.parse_args()

    import torchdtw

    location = Path(torchdtw.__file__).resolve().parent
    if args.expect_dir is not None and not location.is_relative_to(args.expect_dir.resolve()):
        msg = f"The {args.side} imported torchdtw from {location}, not from {args.expect_dir}"
        raise SystemExit(msg)
    pkg_version = _installed_version()
    print(f"{args.side}: torchdtw {pkg_version} from {location}", flush=True)
    outputs, timings = measure_all(
        f"{args.side} {pkg_version}",
        timer=args.timer,
        only=args.only,
        min_run_time=args.min_run_time,
        seed=args.seed,
    )
    dump = {"version": pkg_version, "outputs": outputs, "timings": timings}
    args.output.write_bytes(pickle.dumps(dump))


# --- Driver: install both sides, measure each in a child, compare ---


def _install(spec: str, target: Path) -> None:
    """Install the torchdtw of ``spec`` (a wheel path or a version spec), without dependencies, into ``target``."""
    cmd = ["uv", "pip", "install", "--no-deps", "--target", str(target), "--python", sys.executable, spec]
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _measure(
    side: str,
    target: Path | None,
    output: Path,
    *,
    timer: str,
    only: str | None,
    min_run_time: float,
    seed: int,
) -> dict:
    """Measure one side in a child process: this interpreter, with ``target`` first on ``PYTHONPATH``."""
    cmd = [sys.executable, str(Path(__file__).resolve())]
    cmd += ["--output", str(output), "--side", side, "--min-run-time", str(min_run_time), "--seed", str(seed)]
    cmd += ["--timer", timer]
    if only is not None:
        cmd += ["--only", only]
    env = dict(os.environ)
    if target is not None:
        cmd += ["--expect-dir", str(target)]
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(target), env.get("PYTHONPATH")) if p)
        print(f"PYTHONPATH={shlex.quote(env['PYTHONPATH'])} {shlex.join(cmd)}", flush=True)
    else:
        print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    return pickle.loads(output.read_bytes())


def _changes(baseline: list[Measurement], candidate: list[Measurement]) -> list[tuple[float, str]]:
    """Relative change of the median of each case, from the baseline to the candidate."""
    medians = {(m.task_spec.label, m.task_spec.sub_label): m.median for m in baseline}
    return [
        (m.median / medians[key] - 1, f"{key[0]} {key[1]}")
        for m in candidate
        if (key := (m.task_spec.label, m.task_spec.sub_label)) in medians
    ]


def driver(
    baseline: str | None = None,
    candidate: str = INSTALLED,
    *,
    timer: str = "wall",
    only: str | None = None,
    min_run_time: float = 1,
    seed: int = 0,
) -> None:
    """Measure the baseline and the candidate on the same cases, check that they agree, and compare their times."""
    if baseline is None:
        baseline = f"torchdtw=={latest_version()}"
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for side, spec in (("baseline", baseline), ("candidate", candidate)):
            target = None
            if spec != INSTALLED:
                target = Path(tmp) / side
                _install(spec, target)
            results[side] = _measure(
                side,
                target,
                Path(tmp) / f"{side}.pkl",
                timer=timer,
                only=only,
                min_run_time=min_run_time,
                seed=seed,
            )
    base, cand = results["baseline"], results["candidate"]
    if not base["timings"]:
        msg = f"No case matches --only {only!r}"
        raise SystemExit(msg)

    print(f"\nBaseline:  torchdtw {base['version']} ({baseline})")
    print(f"Candidate: torchdtw {cand['version']} ({candidate})")
    print(f"Timer: {timer}" + (" (CUDA cases only; CPU cases are wall-clock)" if timer == "gpu" else ""))

    common = [key for key in base["outputs"] if key in cand["outputs"]]
    mismatches = []
    for key in common:
        try:
            torch.testing.assert_close(cand["outputs"][key], base["outputs"][key])
        except AssertionError as e:
            mismatches.append(f"  {' '.join(key)}: {e.args[0].splitlines()[0] if e.args else 'diff'}")
    print(f"Correctness: {len(common) - len(mismatches)}/{len(common)} cases match")
    for line in mismatches:
        print(line)

    print()
    compare = Compare(base["timings"] + cand["timings"])
    compare.colorize(rowwise=True)
    compare.print()

    moved = sorted((c for c in _changes(base["timings"], cand["timings"]) if abs(c[0]) > THRESHOLD), reverse=True)
    if moved:
        print(f"Medians beyond ±{THRESHOLD:.0%}, candidate vs baseline:")
        for change, case in moved:
            print(f"  {change:+7.1%}  {case}")
    else:
        print(f"Every median is within ±{THRESHOLD:.0%}.")

    if mismatches:
        print(f"\n{len(mismatches)} correctness mismatch(es).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _child_main()
