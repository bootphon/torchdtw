"""Render benchmark measurements, and inject them into a markdown file between a runner's markers."""

import contextlib
import io
import platform
import re
from pathlib import Path

import torch
from torch.utils.benchmark import Compare, Measurement


def device_name() -> str:
    """Name of the accelerator the benchmark runs on, falling back to the CPU."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine() or "CPU"


def show(results: list[Measurement]) -> None:
    """Print the measurements as tables, colorized across the implementations of each row."""
    compare = Compare(results)
    compare.colorize(rowwise=True)
    compare.print()


def render(results: list[Measurement]) -> str:
    """Render measurements as the plain (uncolored) table text that ``Compare`` prints."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        Compare(results).print()
    return buffer.getvalue().strip("\n")


def inject(output: Path, results: list[Measurement], markers: tuple[str, str]) -> None:
    """Inject the rendered results between the ``markers`` of the ``output`` markdown file."""
    begin, end = markers
    block = f"### Results on {device_name()}\n\n```\n{render(results)}\n```"
    content = output.read_text(encoding="utf-8")
    pattern = re.escape(begin) + r".*?" + re.escape(end)
    new_content, count = re.subn(pattern, lambda _: f"{begin}\n{block}\n{end}", content, flags=re.DOTALL)
    if count == 0:
        msg = f"Markers not found in {output}. Add:\n{begin}\n{end}"
        raise RuntimeError(msg)
    output.write_text(new_content, encoding="utf-8")
