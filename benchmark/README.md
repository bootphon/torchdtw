# DTW benchmark

Small benchmark to compare various exact DTW implementations.
We verify that all outputs are identical.
Submit a PR if you want to add another implementation!

Available implementations:
- PyTorch naive: inefficient PyTorch implementation without any parallelization.
- Cython: convert to numpy array, cython backend for cost computation, then back to PyTorch.
- Numba: adapted from Whisper.
- Triton: adapter from Whisper, CUDA only.
- PyTorch C++ extension: functions from `torchdtw`.

Run with:
```bash
python -m dtw_benchmark
```

Computation time for DTW on array of shape (n, n), on one A40 GPU:

```
[--------------------------------------- cpu ----------------------------------------]
                             |    16    |   32   |   64   |  128   |   256   |   512
10 threads: --------------------------------------------------------------------------
      PyTorch naive          |  3780.3  |        |        |        |         |
      Cython                 |     4.6  |   6.7  |  17.5  |  63.3  |  253.6  |  1027.2
      Numba                  |    14.1  |  15.9  |  23.0  |  69.7  |  807.0  |  3612.1
      PyTorch C++ extension  |     2.9  |   3.9  |   7.5  |  22.9  |   87.4  |   342.6

Times are in microseconds (us).

[----------------------------------------- cuda -----------------------------------------]
                             |     16    |    32   |    64   |   128   |   256   |   512
10 threads: ------------------------------------------------------------------------------
      PyTorch naive          |  15013.3  |         |         |         |         |
      Cython                 |     33.7  |   36.4  |   48.9  |  100.9  |  310.8  |  1153.0
      Numba                  |     53.2  |   54.9  |   65.9  |  124.7  |  879.6  |  3770.8
      Triton                 |    233.8  |  243.8  |  310.1  |  499.9  |  866.1  |  1921.5
      PyTorch C++ extension  |     29.3  |   34.6  |   63.0  |  120.1  |  269.4  |   745.3

Times are in microseconds (us).
```
