# torchdtw benchmarks

Benchmarks of torchdtw against other implementations, and a tool to compare two torchdtw builds.
Every measurement first checks that the outputs agree.
Submit a PR if you want to add another implementation!

Each runner covers the CPU and, when available, CUDA. Pass `--output README.md` to inject its results
into its markers below, and `--min-run-time` to control how long each measurement runs.

## DTW

```bash
python -m dtw_benchmark run dtw
```

Available implementations:
- PyTorch naive: inefficient PyTorch implementation without any parallelization, run on the smallest
  shape only.
- Cython: convert to numpy array, cython backend for cost computation, then back to PyTorch.
- Numba: adapted from Whisper.
- Triton: adapted from Whisper, CUDA only.
- PyTorch C++ extension: functions from `torchdtw`.

The single-pair function (`dtw`) and the batched function (`dtw_batch`) are both benchmarked.
The Cython, Numba and Triton backends provide batched variants too;
the C++ extension is used as the correctness reference for every measurement.

Each batch config runs at three spreads of sequence lengths: "none" (every sequence has the padded
length), "small" (lengths down to 90% of it) and "large" (down to 25%). The lengths are `int32`, the
dtype of the main workload. Each table has one row per case and one column per implementation.

The Cython and Numba backends have no GPU code path: even for a CUDA input they copy the distances
to host, compute on the CPU, and copy the result back, so their `cuda` timings include the
host-device transfers rather than measuring on-device work.

<!-- benchmark -->
### Results on NVIDIA H100 NVL

```
[--------------------------------- dtw / cpu ---------------------------------]
               |  PyTorch naive  |  Cython  |  Numba   |  PyTorch C++ extension
32 threads: -------------------------------------------------------------------
      8x8      |      609.1      |    3.1   |     3.0  |            2.6        
      16x16    |                 |    3.4   |     3.5  |            2.8        
      32x32    |                 |    5.2   |     4.7  |            3.5        
      64x64    |                 |   14.1   |     9.6  |            6.8        
      128x128  |                 |   55.7   |    30.8  |           22.4        
      256x256  |                 |  229.7   |   143.2  |           85.5        
      512x512  |                 |  936.2   |  3782.4  |          338.6        
      128x512  |                 |  237.0   |   200.3  |           87.1        
      512x128  |                 |  216.2   |   157.4  |           81.3        

Times are in microseconds (us).

[-------------------------------------- dtw / cuda --------------------------------------]
               |  PyTorch naive  |  Cython  |  Numba   |  Triton  |  PyTorch C++ extension
32 threads: ------------------------------------------------------------------------------
      8x8      |      2310.6     |    20.8  |    20.8  |   136.5  |           15.7        
      16x16    |                 |    21.4  |    21.4  |   138.4  |           15.7        
      32x32    |                 |    23.9  |    23.1  |   175.3  |           26.8        
      64x64    |                 |    33.7  |    30.0  |   239.3  |           49.2        
      128x128  |                 |    78.7  |    57.2  |   476.5  |           94.7        
      256x256  |                 |   265.4  |   315.4  |   770.8  |          192.3        
      512x512  |                 |  1019.0  |  3938.6  |  1518.9  |          419.5        
      128x512  |                 |   275.1  |   231.0  |   880.4  |          229.6        
      512x128  |                 |   252.4  |   206.7  |  1041.0  |          228.5        

Times are in microseconds (us).

[-------------------------------- dtw_batch / cpu --------------------------------]
                                    |   Cython  |  Numba   |  PyTorch C++ extension
32 threads: -----------------------------------------------------------------------
      n=8x8 s=32x32 sym none        |     69.4  |    91.9  |            15.4       
      n=8x8 s=32x32 sym small       |     63.7  |    91.1  |            15.9       
      n=8x8 s=32x32 sym large       |     33.1  |    87.7  |            15.5       
      n=8x8 s=32x32 asym none       |    149.5  |   107.5  |            25.8       
      n=8x8 s=32x32 asym small      |    136.2  |   105.9  |            25.0       
      n=8x8 s=32x32 asym large      |     65.5  |    95.4  |            19.4       
      n=16x16 s=8x8 sym none        |     20.6  |    91.7  |            13.8       
      n=16x16 s=8x8 sym small       |     19.2  |    90.3  |            14.6       
      n=16x16 s=8x8 sym large       |     13.8  |    89.4  |            12.9       
      n=16x16 s=8x8 asym none       |     36.9  |    95.0  |            16.1       
      n=16x16 s=8x8 asym small      |     34.4  |    83.1  |            15.3       
      n=16x16 s=8x8 asym large      |     20.0  |    93.8  |            15.8       
      n=16x16 s=32x32 sym none      |    275.1  |   134.5  |            20.5       
      n=16x16 s=32x32 sym small     |    252.4  |   132.4  |            19.8       
      n=16x16 s=32x32 sym large     |     95.0  |   103.4  |            18.3       
      n=16x16 s=32x32 asym none     |    581.5  |   179.1  |            37.6       
      n=16x16 s=32x32 asym small    |    530.5  |   175.2  |            36.4       
      n=16x16 s=32x32 asym large    |    212.9  |   139.8  |            27.3       
      n=32x32 s=32x32 sym none      |   1119.6  |   277.5  |            41.5       
      n=32x32 s=32x32 sym small     |   1006.5  |   262.4  |            39.3       
      n=32x32 s=32x32 sym large     |    545.6  |   216.8  |            33.5       
      n=32x32 s=32x32 asym none     |   2311.9  |   428.7  |            60.6       
      n=32x32 s=32x32 asym small    |   2088.7  |   409.5  |            57.7       
      n=32x32 s=32x32 asym large    |    940.7  |   295.4  |            41.5       
      n=64x64 s=16x16 sym none      |   1054.9  |   329.2  |            55.7       
      n=64x64 s=16x16 sym small     |    881.6  |   335.5  |            53.0       
      n=64x64 s=16x16 sym large     |    503.7  |   242.6  |            47.2       
      n=128x128 s=16x16 sym none    |   4545.9  |   841.2  |           169.9       
      n=128x128 s=16x16 sym small   |   3921.4  |   802.7  |           156.1       
      n=128x128 s=16x16 sym large   |   2165.9  |   545.4  |           120.5       
      n=128x128 s=16x16 asym none   |   9145.4  |  1454.7  |           240.1       
      n=128x128 s=16x16 asym small  |   8027.0  |  1373.2  |           214.3       
      n=128x128 s=16x16 asym large  |   4399.0  |   936.9  |           147.2       
      n=256x256 s=8x8 sym none      |   4687.4  |  1159.8  |           321.7       
      n=256x256 s=8x8 sym small     |   4594.9  |  1175.8  |           311.3       
      n=256x256 s=8x8 sym large     |   2991.8  |   890.3  |           252.4       
      n=16x16 s=128x128 sym none    |   6289.5  |   682.2  |           108.8       
      n=16x16 s=128x128 sym small   |   5760.3  |   643.9  |           104.6       
      n=16x16 s=128x128 sym large   |   1884.3  |   385.8  |            61.3       
      n=4x8 s=256x512 asym none     |  14901.5  |  1583.2  |          1379.8       
      n=4x8 s=256x512 asym small    |  13747.0  |  1408.4  |          1303.6       
      n=4x8 s=256x512 asym large    |   4938.4  |   801.3  |           614.5       

Times are in microseconds (us).

[-------------------------------------- dtw_batch / cuda -------------------------------------]
                                    |   Cython  |  Numba   |   Triton  |  PyTorch C++ extension
32 threads: -----------------------------------------------------------------------------------
      n=8x8 s=32x32 sym none        |    128.3  |   162.4  |    357.5  |           26.0        
      n=8x8 s=32x32 sym small       |    121.1  |   165.2  |    365.2  |           26.0        
      n=8x8 s=32x32 sym large       |     98.9  |   157.3  |    356.6  |           23.8        
      n=8x8 s=32x32 asym none       |    213.0  |   181.9  |    418.8  |           24.6        
      n=8x8 s=32x32 asym small      |    199.9  |   181.8  |    419.6  |           23.9        
      n=8x8 s=32x32 asym large      |    147.6  |   170.0  |    410.9  |           22.6        
      n=16x16 s=8x8 sym none        |     65.5  |   143.7  |    329.5  |            9.5        
      n=16x16 s=8x8 sym small       |     64.5  |   144.3  |    341.6  |            9.5        
      n=16x16 s=8x8 sym large       |     60.4  |   143.9  |    341.0  |            9.5        
      n=16x16 s=8x8 asym none       |     80.3  |   148.1  |    378.5  |            6.8        
      n=16x16 s=8x8 asym small      |     78.1  |   149.1  |    384.0  |            6.7        
      n=16x16 s=8x8 asym large      |     64.1  |   147.6  |    390.1  |            6.7        
      n=16x16 s=32x32 sym none      |    402.4  |   249.5  |    428.7  |           26.5        
      n=16x16 s=32x32 sym small     |    372.3  |   248.9  |    426.8  |           26.0        
      n=16x16 s=32x32 sym large     |    235.4  |   222.4  |    427.3  |           25.5        
      n=16x16 s=32x32 asym none     |    721.2  |   301.6  |    502.6  |           24.8        
      n=16x16 s=32x32 asym small    |    665.6  |   296.9  |    511.1  |           24.5        
      n=16x16 s=32x32 asym large    |    414.3  |   262.5  |    509.3  |           23.8        
      n=32x32 s=32x32 sym none      |   1407.2  |   515.9  |    555.0  |           26.7        
      n=32x32 s=32x32 sym small     |   1282.2  |   500.6  |    566.2  |           26.5        
      n=32x32 s=32x32 sym large     |    951.9  |   451.0  |    565.2  |           25.8        
      n=32x32 s=32x32 asym none     |   2623.5  |   675.5  |    730.6  |           25.5        
      n=32x32 s=32x32 asym small    |   2441.9  |   660.9  |    778.8  |           25.3        
      n=32x32 s=32x32 asym large    |   1573.9  |   551.2  |    731.9  |           24.6        
      n=64x64 s=16x16 sym none      |   1314.5  |   572.7  |    716.3  |           16.2        
      n=64x64 s=16x16 sym small     |   1133.4  |   559.7  |    729.3  |           16.0        
      n=64x64 s=16x16 sym large     |    818.4  |   507.5  |    733.4  |           15.0        
      n=128x128 s=16x16 sym none    |   5616.3  |  1621.0  |  33258.0  |           32.7        
      n=128x128 s=16x16 sym small   |   4893.9  |  1807.6  |  30079.5  |           31.1        
      n=128x128 s=16x16 sym large   |   3729.3  |  1447.9  |  32464.1  |           26.1        
      n=128x128 s=16x16 asym none   |  10092.3  |  2357.4  |  27746.6  |           57.0        
      n=128x128 s=16x16 asym small  |   9052.1  |  2328.7  |  35205.6  |           53.9        
      n=128x128 s=16x16 asym large  |   5849.7  |  1761.5  |  39422.3  |           39.5        
      n=256x256 s=8x8 sym none      |   5893.6  |  1926.4  |  39251.2  |           54.7        
      n=256x256 s=8x8 sym small     |   5288.9  |  1928.0  |  49938.6  |           51.6        
      n=256x256 s=8x8 sym large     |   3790.6  |  1704.4  |  21424.9  |           36.4        
      n=16x16 s=128x128 sym none    |   7039.2  |  1780.6  |    841.8  |           97.1        
      n=16x16 s=128x128 sym small   |   6540.2  |  1416.2  |    927.0  |           96.7        
      n=16x16 s=128x128 sym large   |   2661.3  |  1476.8  |    881.6  |           84.9        
      n=4x8 s=256x512 asym none     |  15943.4  |  2369.7  |   1582.6  |          295.4        
      n=4x8 s=256x512 asym small    |  14490.8  |  2405.9  |   1824.4  |          295.2        
      n=4x8 s=256x512 asym large    |   5812.8  |  1538.0  |   1523.8  |          221.4        

Times are in microseconds (us).
```
<!-- /benchmark -->


## Comparing two torchdtw builds

```bash
python -m dtw_benchmark compare [BASELINE] [CANDIDATE] [--timer {wall,gpu}] [--only SUBSTRING] [--min-run-time S]
```

`BASELINE` and `CANDIDATE` are each a wheel path, a version spec (`torchdtw==1.2.3`), or `installed`,
the torchdtw of the current environment. By default, the latest release on PyPI is compared against
`installed`.

- `--timer wall` (default): wall-clock time per call, with a CUDA sync after each call, so that it
  includes the host overhead.
- `--timer gpu`: GPU time per call, from CUDA events around a loop of calls. CPU cases stay wall-clock.
- `--only`: keep the cases whose label contains the substring, e.g. `--only "cuda n=16x16 s=32x32"`,
  to re-run only the cases that moved.
