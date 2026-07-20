"""Import smoke test against the oldest supported PyTorch."""
# ruff: noqa: T201

import sys
from importlib.metadata import version

import torch

import torchdtw

EXPECTED_FLOOR = (2, 10)  # Keep in sync with TORCH_TARGET_VERSION in setup.py


def main() -> int:
    """Exercise every backend on each available device."""
    if tuple(int(part) for part in torch.__version__.split(".")[:2]) != EXPECTED_FLOOR:
        print(f"WARNING: torch {torch.__version__} is not the {EXPECTED_FLOOR} floor this test targets")
    print(f"imported torchdtw {version('torchdtw')} against torch {torch.__version__}")
    devices = ["cpu"]
    if torch.cuda.is_available() and sys.platform != "win32":
        devices.append("cuda")
    for device in devices:
        cost = torchdtw.dtw(torch.rand(8, 8, device=device))
        assert cost.shape == (), cost.shape
        assert torch.isfinite(cost), cost
        lengths = torch.full((2,), 8, dtype=torch.int64, device=device)
        batch = torchdtw.dtw_batch(torch.rand(2, 2, 8, 8, device=device), lengths, lengths, symmetric=False)
        assert batch.shape == (2, 2), batch.shape
        assert torch.isfinite(batch).all(), batch
        print(f"  {device}: dtw and dtw_batch ok")
    path = torchdtw.dtw_path(torch.rand(8, 8))
    assert path.ndim == 2, path.shape
    assert path.shape[1] == 2, path.shape
    print("  cpu: dtw_path ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
