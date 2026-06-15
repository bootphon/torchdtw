"""The CUDA int64-indexed path: nx * ny * max_x * max_y > INT32_MAX.

dtw_batch_cuda switches the distances accessor from int32 to int64 element indexing once the
tensor has more than 2^31 elements (see ``needs_64bit`` in csrc/cuda/dtw.cu). The int32 path
would overflow the element offset and read the wrong memory for high-index pairs. Nothing else
in the suite crosses that threshold, so this whole branch is otherwise dead.

To trigger it the tensor must genuinely hold > 2^31 *elements* -- a strided/broadcast view keeps
the real offsets small and would not exercise the overflow. The smallest such tensor uses a
1-byte dtype (uint8), so the allocation is ~2.0 GB, the floor for honestly crossing the boundary.
Only the corner pairs are checked (against the CPU kernel), including the last pair whose element
offset is past 2^31; the test skips if the GPU lacks the free memory.
"""

import pytest
import torch

from torchdtw import dtw, dtw_batch

from .conftest import assert_equal

INT32_MAX = 2**31 - 1


@pytest.mark.requires_gpu
def test_dtw_batch_cuda_64bit_indexing() -> None:
    """dtw_batch reads the right memory for pairs whose element offset exceeds 2^31."""
    s = 16
    ny = 128
    block = ny * s * s
    nx = INT32_MAX // block + 2  # smallest nx with nx * ny * s * s > INT32_MAX
    n_elem = nx * ny * s * s
    assert n_elem > INT32_MAX

    needed = n_elem + nx * ny  # distances (uint8) + output (uint8), in bytes
    free, _ = torch.cuda.mem_get_info()
    headroom = 1 << 30  # keep ~1 GiB spare
    if free < needed + headroom:
        pytest.skip(f"need ~{needed / 1e9:.1f} GB free GPU memory, have {free / 1e9:.1f} GB")

    g = torch.Generator(device="cuda").manual_seed(0)
    d = torch.randint(0, 4, (nx, ny, s, s), generator=g, dtype=torch.uint8, device="cuda")
    sx = torch.full((nx,), s, dtype=torch.long, device="cuda")
    sy = torch.full((ny,), s, dtype=torch.long, device="cuda")

    out = dtw_batch(d, sx, sy, symmetric=False)

    # The last pair's data sits past the 2^31-element boundary, exactly where the int32 path
    # would overflow. The CPU kernel shares the integer floor-division semantics, so the
    # comparison is exact.
    for i, j in [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]:
        expected = dtw(d[i, j].cpu())
        assert_equal(out[i, j].cpu(), expected)
