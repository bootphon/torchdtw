"""Half / bfloat16 accuracy: reduced-precision output tracks the float32 result.

The DP cost is accumulated in float for half and bfloat16 (see csrc/acc_type.h), so a reduced-precision
call runs the same float recurrence as a float32 call on the same (losslessly up-cast) inputs and only
rounds the final value to the storage dtype. The reduced-precision output therefore stays accurate to
the dtype's own precision instead of drifting as a naive scalar_t accumulation would on long sequences.
"""

import pytest
import torch
from hypothesis import given

from torchdtw import dtw, dtw_batch

from .conftest import BATCH, DIM, HIGH_MINUS_LOW, LOW, make_tensor

ACC_DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", ACC_DTYPES)
@given(x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_half_accuracy(dtype: torch.dtype, x: int, y: int, low: float, high_minus_low: float) -> None:
    """Dtw in half/bfloat16 stays accurate to the float32 result at the dtype's precision."""
    d = make_tensor((x, y), dtype=dtype, low=low, high=high_minus_low + low)
    torch.testing.assert_close(dtw(d), dtw(d.float()).to(dtype))


@pytest.mark.parametrize("dtype", ACC_DTYPES)
@given(n=BATCH, m=BATCH, x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_dtw_batch_half_accuracy(
    dtype: torch.dtype, n: int, m: int, x: int, y: int, low: float, high_minus_low: float
) -> None:
    """dtw_batch in half/bfloat16 stays accurate to the float32 result at the dtype's precision."""
    d = make_tensor((n, m, x, y), dtype=dtype, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x + 1)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y + 1)
    out = dtw_batch(d, sx, sy, symmetric=False)
    reference = dtw_batch(d.float(), sx, sy, symmetric=False).to(dtype)
    torch.testing.assert_close(out, reference)
