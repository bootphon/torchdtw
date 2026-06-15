# DTW benchmark

Small benchmark to compare various exact DTW implementations.
We verify that all outputs are identical.
Submit a PR if you want to add another implementation!

Available implementations:
- PyTorch naive: inefficient PyTorch implementation without any parallelization.
- Cython: convert to numpy array, cython backend for cost computation, then back to PyTorch.
- Numba: adapted from Whisper.
- Triton: adapted from Whisper, CUDA only.
- PyTorch C++ extension: functions from `torchdtw`.

The single-pair function (`dtw`) and the batched function (`dtw_batch`) are both benchmarked.
The Cython, Numba and Triton backends provide batched variants too;
the C++ extension is used as the correctness reference for every measurement.

Run with:
```bash
python -m dtw_benchmark run
```

Pass `--output README.md` to inject the results into the benchmark markers below, and
`--min-run-time` to control how long each measurement runs. Use `python -m dtw_benchmark compare`
to instead compare the current checkout against the latest released torchdtw.

The Cython and Numba backends have no GPU code path: even for a CUDA input they copy the distances
to host, compute on the CPU, and copy the result back, so their `cuda` timings include the
host-device transfers rather than measuring on-device work.

<!-- benchmark -->
## Benchmark results on NVIDIA H100 NVL

```
[------------------------------------------------------- cpu --------------------------------------------------------]
                             |  16x16   |  32x32  |  64x64  |  128x128  |  256x256  |  512x512  |  128x512  |  512x128
10 threads: ----------------------------------------------------------------------------------------------------------
      PyTorch naive          |  2755.6  |         |         |           |           |           |           |         
      Cython                 |     3.4  |   5.2   |   14.6  |    56.8   |   231.7   |    943.0  |   238.9   |   220.3 
      Numba                  |     3.5  |   4.7   |    9.5  |    31.3   |   144.6   |   3868.2  |   191.5   |   158.4 
      PyTorch C++ extension  |     2.7  |   3.4   |    6.8  |    21.8   |    85.4   |    338.5  |    87.4   |    80.7 

Times are in microseconds (us).

[-------------------------------------------------------- cuda -------------------------------------------------------]
                             |   16x16   |  32x32  |  64x64  |  128x128  |  256x256  |  512x512  |  128x512  |  512x128
10 threads: -----------------------------------------------------------------------------------------------------------
      PyTorch naive          |  10069.8  |         |         |           |           |           |           |         
      Cython                 |     22.5  |   25.3  |   35.2  |    81.1   |   270.7   |   1020.4  |   276.4   |    253.5
      Numba                  |     22.5  |   24.2  |   30.8  |    58.8   |   360.7   |   2678.5  |   223.5   |    210.0
      Triton                 |    146.3  |  179.5  |  243.4  |   428.6   |   771.4   |   1478.8  |   888.2   |   1040.9
      PyTorch C++ extension  |     16.3  |   27.3  |   49.8  |    95.5   |   193.0   |    420.5  |   230.1   |    228.8

Times are in microseconds (us).

[---------------------------------------------------------------------------------- cpu (batch) ----------------------------------------------------------------------------------]
                             |  n=16x16 s=128x128 sym  |  n=32x32 s=64x64 asym  |  n=64x64 s=32x32 sym  |  n=128x128 s=16x16 sym  |  n=128x128 s=16x16 asym  |  n=256x256 s=8x8 sym
10 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Cython                 |          1200.7         |         3136.7         |         1750.2        |          1657.0         |          3700.9          |         2781.9      
      Numba                  |           661.1         |         1395.0         |          953.5        |           907.0         |          1779.8          |         1533.5      
      PyTorch C++ extension  |           101.1         |          328.1         |          154.3        |           171.7         |           383.5          |          373.0      

Times are in microseconds (us).

[---------------------------------------------------------------------------------- cuda (batch) ---------------------------------------------------------------------------------]
                             |  n=16x16 s=128x128 sym  |  n=32x32 s=64x64 asym  |  n=64x64 s=32x32 sym  |  n=128x128 s=16x16 sym  |  n=128x128 s=16x16 asym  |  n=256x256 s=8x8 sym
10 threads: -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
      Cython                 |          2025.7         |         4313.0         |         2865.1        |          2756.5         |          5101.3          |         3536.8      
      Numba                  |          1466.7         |         2260.7         |         1789.6        |          1776.7         |          2787.8          |         2448.3      
      Triton                 |           863.4         |         1445.7         |         1025.1        |          6338.3         |          6841.4          |         5036.9      
      PyTorch C++ extension  |            87.9         |           48.7         |           27.2        |            24.8         |            35.1          |           34.5      

Times are in microseconds (us).
```
<!-- /benchmark -->
