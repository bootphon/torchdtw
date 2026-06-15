# ruff: noqa: S603, T201
"""Compare dtw and dtw_batch between the current checkout and the latest released torchdtw.

Driven by the ``compare`` subcommand of ``python -m dtw_benchmark`` (the CLI lives in ``__main__``).
``driver`` spawns a child via ``uv run --with torchdtw==<ver>`` to measure the released version with
identical inputs, runs the local version the same way, then checks correctness and prints a
side-by-side timing comparison.

Each child runs this module directly as a script (``uv run ... python compare_release.py ...``) in a
clean environment that only has ``torch`` and ``torchdtw`` installed. Running it as a script puts its
own directory on ``sys.path``, so it only depends on ``torch`` and ``torchdtw`` -- not the rest of the
``dtw_benchmark`` package, which would not be importable there.
"""

import argparse
import json
import pickle
import shlex
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from packaging.utils import parse_wheel_filename
from torch.utils.benchmark import Compare, Measurement, Timer

import torchdtw

DTW_SHAPES = [
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
]
BATCH_CONFIGS = [
    (4, 4, 128, 128, True),
    (4, 4, 128, 128, False),
    (8, 8, 256, 256, True),
    (4, 8, 256, 512, False),
    (64, 64, 8, 8, True),
    (32, 32, 16, 16, True),
    (16, 32, 32, 32, False),
]


def latest_version() -> str:
    """Query PyPI to get the latest torchdtw version."""
    cmd = ["uv", "run", "--with", "pip", "pip", "index", "versions", "torchdtw", "--json"]
    process = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(process.stdout)["latest"]


def measure_dtw(
    shape: tuple[int, int],
    label: str,
    *,
    device: torch.device,
    min_run_time: float,
    seed: int,
) -> tuple[torch.Tensor, Measurement]:
    """Measure the 'dtw' function and return the output on a random sample."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand(shape, generator=g, dtype=torch.float32).to(device)
    out = torchdtw.dtw(x).detach().cpu().clone()
    sync = "; torch.cuda.synchronize()" if x.is_cuda else ""
    t = Timer(
        stmt=f"dtw(x){sync}",
        setup="from torchdtw import dtw\nimport torch",
        globals={"x": x},
        label=f"dtw / {device.type}",
        sub_label=f"{shape[0]}x{shape[1]}",
        description=label,
    ).blocked_autorange(min_run_time=min_run_time)
    return out, t


def measure_batch(
    cfg: tuple[int, int, int, int, bool],
    label: str,
    *,
    device: torch.device,
    num_threads: int,
    min_run_time: float,
    seed: int,
) -> tuple[torch.Tensor, Measurement]:
    """Measure the 'dtw_batch' function and return the output on a random sample."""
    n1, n2, s1, s2, symmetric = cfg
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.rand((n1, n2, s1, s2), generator=g, dtype=torch.float32)
    if symmetric:
        n = min(n1, n2)
        i, j = torch.triu_indices(n, n)
        d[:n, :n][i, j] = d[:n, :n][j, i]
    sx = torch.randint(1, s1 + 1, (n1,), generator=g, dtype=torch.long)
    sy = sx.clone() if symmetric and n1 == n2 else torch.randint(1, s2 + 1, (n2,), generator=g, dtype=torch.long)
    d, sx, sy = d.to(device), sx.to(device), sy.to(device)

    out = torchdtw.dtw_batch(d, sx, sy, symmetric=symmetric).detach().cpu().clone()
    sync = "; torch.cuda.synchronize()" if d.is_cuda else ""
    threads_label = f", t={num_threads}" if device.type == "cpu" else ""
    t = Timer(
        stmt=f"dtw_batch(d, sx, sy, symmetric=sym){sync}",
        setup="from torchdtw import dtw_batch\nimport torch",
        globals={"d": d, "sx": sx, "sy": sy, "sym": symmetric},
        num_threads=num_threads,
        label=f"dtw_batch / {device.type}",
        sub_label=f"n={n1}x{n2}, s={s1}x{s2}, {'sym' if symmetric else 'asym'}{threads_label}",
        description=label,
    ).blocked_autorange(min_run_time=min_run_time)
    return out, t


def measure_all(label: str, *, min_run_time: float, seed: int) -> tuple[dict, list[Measurement]]:
    """Measure 'dtw' and 'dtw_batch' on all configurations."""
    outputs, timings = {}, []
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for device in devices:
        thread_counts = sorted({1, torch.get_num_threads()}) if device.type == "cpu" else [1]
        for shape in DTW_SHAPES:
            out, t = measure_dtw(shape, label, device=device, min_run_time=min_run_time, seed=seed)
            outputs[("dtw", device.type, shape)] = out
            timings.append(t)
        for cfg in BATCH_CONFIGS:
            for num_threads in thread_counts:
                out, t = measure_batch(
                    cfg,
                    label,
                    device=device,
                    num_threads=num_threads,
                    min_run_time=min_run_time,
                    seed=seed,
                )
                outputs[("dtw_batch", device.type, cfg)] = out
                timings.append(t)
    return outputs, timings


def measure_and_dump(label: str, output: str | Path, *, min_run_time: float, seed: int) -> None:
    """Measure 'dtw' and 'dtw_batch' and dump the output with pickle."""
    try:
        pkg_version = version("torchdtw")
    except PackageNotFoundError:
        pkg_version = "unknown"
    outputs, timings = measure_all(label, min_run_time=min_run_time, seed=seed)
    dump = pickle.dumps({"label": label, "version": pkg_version, "outputs": outputs, "timings": timings})
    Path(output).write_bytes(dump)


def _measure_command(uv_args: list[str], *, output: Path, label: str, min_run_time: float, seed: int) -> list[str]:
    """Build the ``uv run ... python compare_release.py`` command that measures one torchdtw version."""
    child = [
        "python",
        str(Path(__file__)),
        "--output",
        str(output),
        "--label",
        label,
        "--min-run-time",
        str(min_run_time),
        "--seed",
        str(seed),
    ]
    return ["uv", "run", *uv_args, *child]


def driver(*, min_run_time: float, seed: int, path_wheel: str | Path | None = None) -> None:
    """Driver for measurements across versions."""
    if path_wheel is None:
        reference_version = latest_version()
        reference = f"torchdtw=={reference_version}"
    else:
        reference_version = str(parse_wheel_filename(Path(path_wheel).name)[1])
        reference = str(path_wheel)

    with tempfile.TemporaryDirectory() as tmp:
        released_pkl = Path(tmp) / "released.pkl"
        released_cmd = _measure_command(
            ["--no-project", "--with", reference],
            output=released_pkl,
            label=f"reference {reference_version}",
            min_run_time=min_run_time,
            seed=seed,
        )
        print(shlex.join(released_cmd))
        subprocess.run(released_cmd, check=True)
        released = pickle.loads(released_pkl.read_bytes())  # noqa: S301

        local_pkl = Path(tmp) / "local.pkl"
        local_cmd = _measure_command([], output=local_pkl, label="current", min_run_time=min_run_time, seed=seed)
        print(shlex.join(local_cmd))
        subprocess.run(local_cmd, check=True)
        local = pickle.loads(local_pkl.read_bytes())  # noqa: S301

    print(f"Reference torchdtw version: {released['version']}")
    print(f"Current torchdtw version:   {local['version']}")
    print("Correctness (current vs reference):")

    mismatches = 0
    for key, ref in released["outputs"].items():
        cur = local["outputs"][key]
        try:
            torch.testing.assert_close(cur, ref)
            status = "OK"
        except AssertionError as e:
            mismatches += 1
            status = f"MISMATCH ({e.args[0].splitlines()[0] if e.args else 'diff'})"
        print(f"  {key}: {status}")

    print()
    compare = Compare(released["timings"] + local["timings"])
    compare.colorize(rowwise=True)
    compare.print()
    if mismatches:
        print(f"\n{mismatches} correctness mismatch(es).", file=sys.stderr)
        sys.exit(1)


def _child_main() -> None:
    """Child entry point: measure one torchdtw version and dump the result. Run by the driver via uv."""
    parser = argparse.ArgumentParser(description="Measure one torchdtw version (internal child of 'compare').")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the pickled measurements to.")
    parser.add_argument("--label", required=True, help="Label identifying this version in the comparison.")
    parser.add_argument("--min-run-time", type=float, required=True, help="min_run_time for blocked_autorange.")
    parser.add_argument("--seed", type=int, required=True, help="Seed for the random input tensors.")
    args = parser.parse_args()
    measure_and_dump(args.label, args.output, min_run_time=args.min_run_time, seed=args.seed)


if __name__ == "__main__":
    _child_main()
