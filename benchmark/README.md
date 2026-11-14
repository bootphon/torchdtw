# torchdtw benchmarks

Benchmarks of torchdtw against other implementations, and a tool to compare two torchdtw builds.
Every measurement first checks that the outputs agree with torchdtw's.
Each implementation runs in its own process, so that the thread pools of one (Numba loads a second
OpenMP runtime next to PyTorch's) cannot slow down another.
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
- torchdtw: `dtw` and `dtw_batch`.

The single-pair function (`dtw`) and the batched function (`dtw_batch`) are both benchmarked.
The Cython, Numba and Triton backends provide batched variants too;
torchdtw is the correctness reference for every measurement.

Each batch config runs at three spreads of sequence lengths: "none" (every sequence has the padded
length), "small" (lengths down to 90% of it) and "large" (down to 25%). The lengths are `int32`, the
dtype of the main workload. Each table has one row per case and one column per implementation.

The Cython and Numba backends have no GPU code path: even for a CUDA input they copy the distances
to host, compute on the CPU, and copy the result back, so their `cuda` timings include the
host-device transfers rather than measuring on-device work.

<!-- benchmark -->
### Results on NVIDIA H100 80GB HBM3

```
[-------------------------- dtw / cpu ---------------------------]
               |  PyTorch naive  |  Cython  |  Numba   |  torchdtw
24 threads: ------------------------------------------------------
      8x8      |      779.5      |    3.5   |     3.5  |     2.8  
      16x16    |                 |    3.8   |     4.1  |     2.8  
      32x32    |                 |    5.6   |     5.6  |     3.5  
      64x64    |                 |   13.2   |    11.2  |     7.4  
      128x128  |                 |   52.1   |    91.2  |    25.0  
      256x256  |                 |  228.4   |   530.7  |   118.0  
      512x512  |                 |  949.9   |  2531.8  |   513.1  
      128x512  |                 |  234.9   |   547.8  |   132.7  
      512x128  |                 |  214.1   |   438.7  |    97.3  

Times are in microseconds (us).

[-------------------------------- dtw / cuda -------------------------------]
               |  PyTorch naive  |  Cython  |  Numba   |  Triton  |  torchdtw
24 threads: -----------------------------------------------------------------
      8x8      |      2856.4     |    24.5  |    23.8  |   114.9  |     9.3  
      16x16    |                 |    24.7  |    23.9  |   122.1  |    11.0  
      32x32    |                 |    27.6  |    26.2  |   146.2  |    21.0  
      64x64    |                 |    38.2  |    36.8  |   207.4  |    40.8  
      128x128  |                 |    99.1  |   125.6  |   366.8  |    80.5  
      256x256  |                 |   275.6  |   530.5  |   699.5  |   166.6  
      512x512  |                 |  1179.7  |  2601.6  |  1417.0  |   365.3  
      128x512  |                 |   311.7  |   501.4  |   768.4  |   198.5  
      512x128  |                 |   287.7  |   463.4  |   908.9  |   199.3  

Times are in microseconds (us).

[------------------------- dtw_batch / cpu --------------------------]
                                    |   Cython  |  Numba   |  torchdtw
24 threads: ----------------------------------------------------------
      n=8x8 s=32x32 sym none        |     74.5  |    37.5  |    12.9  
      n=8x8 s=32x32 sym small       |     68.2  |    36.3  |    13.0  
      n=8x8 s=32x32 sym large       |     36.0  |    32.4  |    11.2  
      n=8x8 s=32x32 asym none       |    160.5  |    42.9  |    14.6  
      n=8x8 s=32x32 asym small      |    143.8  |    40.7  |    14.2  
      n=8x8 s=32x32 asym large      |     69.1  |    38.4  |    12.5  
      n=16x16 s=8x8 sym none        |     24.8  |    34.5  |    11.2  
      n=16x16 s=8x8 sym small       |     22.4  |    33.8  |    11.2  
      n=16x16 s=8x8 sym large       |     16.7  |    34.4  |    10.8  
      n=16x16 s=8x8 asym none       |     43.2  |    37.9  |    11.4  
      n=16x16 s=8x8 asym small      |     41.1  |    37.0  |    10.8  
      n=16x16 s=8x8 asym large      |     22.8  |    35.2  |    10.3  
      n=16x16 s=32x32 sym none      |    287.5  |    50.7  |    18.0  
      n=16x16 s=32x32 sym small     |    260.0  |    49.3  |    17.6  
      n=16x16 s=32x32 sym large     |     97.8  |    40.8  |    14.3  
      n=16x16 s=32x32 asym none     |    633.1  |   127.7  |    27.8  
      n=16x16 s=32x32 asym small    |    572.6  |   118.7  |    25.1  
      n=16x16 s=32x32 asym large    |    225.5  |    54.7  |    18.2  
      n=32x32 s=32x32 sym none      |   1206.5  |   242.3  |    42.9  
      n=32x32 s=32x32 sym small     |   1096.3  |   225.7  |    40.4  
      n=32x32 s=32x32 sym large     |    603.3  |   157.6  |    32.1  
      n=32x32 s=32x32 asym none     |   2510.6  |   484.1  |    77.7  
      n=32x32 s=32x32 asym small    |   2123.5  |   461.7  |    70.4  
      n=32x32 s=32x32 asym large    |   1082.3  |   270.9  |    47.0  
      n=64x64 s=16x16 sym none      |   1164.6  |   297.2  |    48.4  
      n=64x64 s=16x16 sym small     |   1007.3  |   269.6  |    45.0  
      n=64x64 s=16x16 sym large     |    653.5  |   185.9  |    32.2  
      n=128x128 s=16x16 sym none    |   4564.8  |  1137.0  |   197.2  
      n=128x128 s=16x16 sym small   |   4186.6  |  1027.3  |   177.7  
      n=128x128 s=16x16 sym large   |   2593.0  |   612.2  |   117.5  
      n=128x128 s=16x16 asym none   |   9824.8  |  2236.7  |   360.9  
      n=128x128 s=16x16 asym small  |   8598.1  |  2048.1  |   325.6  
      n=128x128 s=16x16 asym large  |   4598.4  |  1213.3  |   204.2  
      n=256x256 s=8x8 sym none      |   5269.0  |  1317.9  |   274.5  
      n=256x256 s=8x8 sym small     |   5041.0  |  1263.1  |   273.8  
      n=256x256 s=8x8 sym large     |   3573.9  |   931.6  |   193.9  
      n=16x16 s=128x128 sym none    |   6174.1  |   911.2  |   168.1  
      n=16x16 s=128x128 sym small   |   5696.7  |   860.8  |   168.0  
      n=16x16 s=128x128 sym large   |   1913.4  |   455.9  |    84.2  
      n=4x8 s=256x512 asym none     |  15218.4  |  3490.8  |   622.2  
      n=4x8 s=256x512 asym small    |  13842.3  |  3121.5  |   639.7  
      n=4x8 s=256x512 asym large    |   4987.4  |  1473.4  |   306.9  

Times are in microseconds (us).

[------------------------------- dtw_batch / cuda ------------------------------]
                                    |   Cython  |  Numba   |  Triton  |  torchdtw
24 threads: ---------------------------------------------------------------------
      n=8x8 s=32x32 sym none        |    140.0  |   118.9  |   338.5  |    23.8  
      n=8x8 s=32x32 sym small       |    137.7  |   119.5  |   345.0  |    23.8  
      n=8x8 s=32x32 sym large       |    105.7  |   113.4  |   332.8  |    21.9  
      n=8x8 s=32x32 asym none       |    222.7  |   129.7  |   408.7  |    21.8  
      n=8x8 s=32x32 asym small      |    204.4  |   123.8  |   399.5  |    21.6  
      n=8x8 s=32x32 asym large      |    140.0  |   120.1  |   397.1  |    20.9  
      n=16x16 s=8x8 sym none        |     72.0  |    91.8  |   310.7  |    11.0  
      n=16x16 s=8x8 sym small       |     68.9  |    91.0  |   313.0  |    11.0  
      n=16x16 s=8x8 sym large       |     65.3  |    89.8  |   314.0  |    10.9  
      n=16x16 s=8x8 asym none       |     91.1  |    87.9  |   361.8  |     7.1  
      n=16x16 s=8x8 asym small      |     87.1  |    87.8  |   354.7  |     7.0  
      n=16x16 s=8x8 asym large      |     76.7  |    87.0  |   362.4  |     7.1  
      n=16x16 s=32x32 sym none      |    416.1  |   186.2  |   444.8  |    24.0  
      n=16x16 s=32x32 sym small     |    390.1  |   187.9  |   435.8  |    23.6  
      n=16x16 s=32x32 sym large     |    245.1  |   172.6  |   396.5  |    23.6  
      n=16x16 s=32x32 asym none     |    830.5  |   269.0  |   776.8  |    22.1  
      n=16x16 s=32x32 asym small    |    772.9  |   284.7  |   622.0  |    22.1  
      n=16x16 s=32x32 asym large    |    423.9  |   210.8  |   545.9  |    21.5  
      n=32x32 s=32x32 sym none      |   1522.1  |   606.2  |   754.2  |    24.1  
      n=32x32 s=32x32 sym small     |   1447.5  |   613.1  |   720.5  |    24.1  
      n=32x32 s=32x32 sym large     |   1048.2  |   547.1  |   680.8  |    23.6  
      n=32x32 s=32x32 asym none     |   2890.2  |   834.3  |  1458.0  |    22.9  
      n=32x32 s=32x32 asym small    |   2538.7  |   815.0  |  1203.0  |    22.9  
      n=32x32 s=32x32 asym large    |   1413.5  |   714.2  |  1169.6  |    22.2  
      n=64x64 s=16x16 sym none      |   1462.2  |   692.9  |  1014.4  |    14.7  
      n=64x64 s=16x16 sym small     |   1325.1  |   665.5  |   948.7  |    14.5  
      n=64x64 s=16x16 sym large     |   1000.9  |   563.3  |  1020.2  |    14.0  
      n=128x128 s=16x16 sym none    |   6468.5  |  1976.2  |  2967.9  |    29.2  
      n=128x128 s=16x16 sym small   |   5212.6  |  1918.2  |  2385.0  |    27.8  
      n=128x128 s=16x16 sym large   |   3333.3  |  1557.0  |  2463.2  |    23.3  
      n=128x128 s=16x16 asym none   |  10236.1  |  4018.4  |  5334.8  |    50.8  
      n=128x128 s=16x16 asym small  |  10881.2  |  4303.0  |  4884.6  |    48.0  
      n=128x128 s=16x16 asym large  |   6096.5  |  2944.0  |  4246.1  |    35.2  
      n=256x256 s=8x8 sym none      |   5766.9  |  2703.1  |  4158.7  |    49.4  
      n=256x256 s=8x8 sym small     |   6291.2  |  2457.8  |  3975.1  |    46.6  
      n=256x256 s=8x8 sym large     |   5586.1  |  2042.0  |  3146.7  |    33.2  
      n=16x16 s=128x128 sym none    |   7699.9  |  2059.1  |  1531.4  |    87.0  
      n=16x16 s=128x128 sym small   |   7153.2  |  2567.2  |  1816.9  |    86.3  
      n=16x16 s=128x128 sym large   |   3056.2  |  1826.8  |  1438.7  |    76.7  
      n=4x8 s=256x512 asym none     |  17089.7  |  5744.9  |  2802.3  |   263.7  
      n=4x8 s=256x512 asym small    |  16166.7  |  5400.9  |  4082.9  |   261.4  
      n=4x8 s=256x512 asym large    |   6686.0  |  2593.9  |  3982.1  |   197.5  

Times are in microseconds (us).
```
<!-- /benchmark -->

## Soft-DTW

```bash
python -m dtw_benchmark run soft [--forward-only]
```

Available implementations, each from padded sequences x and y with the squared Euclidean distance:
- [pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) (Maghoumi): Numba on the CPU
  and Numba CUDA kernels on the GPU. Not on PyPI: vendored as is in `soft/soft_dtw_cuda.py`, with its
  MIT license.
- [sdtw-cuda-torch](https://github.com/BGU-CS-VIL/sdtw-cuda-torch) (BGU-CS-VIL): Numba on the CPU, and
  Numba CUDA kernels on the GPU, in two modes. "sdtw-cuda-torch" is its unfused mode: the distances
  come from one `bmm`, then the DP kernels run on them. This is its fast mode, and the only one on the
  CPU. "sdtw-cuda-torch fused" is its default on CUDA: it recomputes the distances inside the DP, with
  one kernel launch per anti-diagonal, and so never stores the (B, N, M) distances, at the cost of speed.
  Not on PyPI: installed from its git repository, at a pinned commit.
- [tslearn](https://github.com/tslearn-team/tslearn): `SoftDTWLossPyTorch`, Numba on the CPU only.
- torchdtw: `soft_dtw` on `torch.cdist(x, y) ** 2`.

Each call is a forward and a backward pass, to the gradient with respect to x, unless `--forward-only`.
Each config (B pairs of sequences of padded length L, with frames of dimension D) runs at the three
spreads of sequence lengths of the DTW benchmark, with `int32` lengths. Only torchdtw takes per-sequence
lengths: the other implementations are called once per pair, each sliced to its lengths, whenever the
lengths of a batch are uneven (the "small" and "large" spreads).

<!-- benchmark-soft -->
### Results on NVIDIA H100 80GB HBM3

```
[----------------------------- soft_dtw / cpu (forward + backward) ------------------------------]
                             |  pytorch-softdtw-cuda  |  sdtw-cuda-torch  |  tslearn   |  torchdtw
24 threads: --------------------------------------------------------------------------------------
      B=64 L=16 D=8 none     |           778.9        |         733.0     |     712.7  |    363.8 
      B=64 L=16 D=8 small    |         17634.2        |       21252.7     |   13738.6  |    342.3 
      B=64 L=16 D=8 large    |         16744.4        |       20031.6     |   12506.1  |    240.2 
      B=64 L=32 D=8 none     |          1275.9        |        1242.8     |    1088.1  |   1001.2 
      B=64 L=32 D=8 small    |         23858.2        |       25293.2     |   19345.5  |    943.5 
      B=64 L=32 D=8 large    |         19605.7        |       22207.7     |   15031.8  |    762.0 
      B=64 L=64 D=8 none     |          3239.5        |        2631.8     |    3289.6  |   2265.4 
      B=64 L=64 D=8 small    |         44800.9        |       42634.9     |   42849.6  |   2204.5 
      B=64 L=64 D=8 large    |         29007.8        |       30168.4     |   25910.3  |   1726.9 
      B=64 L=256 D=12 none   |         74842.2        |       36027.1     |   69751.6  |  24312.9 
      B=64 L=256 D=12 small  |        364477.2        |      347272.6     |  379264.1  |  23503.1 
      B=64 L=256 D=12 large  |        187879.4        |      180657.7     |  191648.9  |  17674.3 
      B=32 L=256 D=16 none   |         45422.2        |       16763.7     |   44430.1  |  16069.0 
      B=32 L=256 D=16 small  |        183315.0        |      175946.9     |  189264.0  |  15418.5 
      B=32 L=256 D=16 large  |         89351.7        |       84575.3     |   90757.8  |  12205.6 
      B=16 L=512 D=16 none   |        102428.5        |       40450.9     |   89276.9  |  31618.9 
      B=16 L=512 D=16 small  |        352299.8        |      326604.7     |  366272.0  |  28973.5 
      B=16 L=512 D=16 large  |        125627.4        |      117384.5     |  130606.6  |  20415.6 

Times are in microseconds (us).

[------------------------------------ soft_dtw / cuda (forward + backward) -----------------------------------]
                             |  pytorch-softdtw-cuda  |  sdtw-cuda-torch  |  sdtw-cuda-torch fused  |  torchdtw
24 threads: ---------------------------------------------------------------------------------------------------
      B=64 L=16 D=8 none     |          901.4         |       1116.2      |           2825.7        |    197.2 
      B=64 L=16 D=8 small    |        43187.6         |      51280.0      |         148016.3        |    191.2 
      B=64 L=16 D=8 large    |        43563.9         |      50718.7      |         114890.4        |    194.5 
      B=64 L=32 D=8 none     |          898.9         |       1096.2      |           4724.2        |    418.6 
      B=64 L=32 D=8 small    |        44160.5         |      53628.7      |         266892.7        |    418.2 
      B=64 L=32 D=8 large    |        40742.4         |      52075.5      |         180554.3        |    418.3 
      B=64 L=64 D=8 none     |          874.6         |       1067.1      |           8323.3        |    416.2 
      B=64 L=64 D=8 small    |        46005.5         |      56729.7      |         483366.7        |    415.7 
      B=64 L=64 D=8 large    |        44811.9         |      55090.5      |         346864.5        |    405.0 
      B=64 L=256 D=12 none   |         2634.6         |       1800.7      |          31428.2        |   1025.2 
      B=64 L=256 D=12 small  |        81365.2         |      82347.1      |        1818901.4        |   1087.8 
      B=64 L=256 D=12 large  |        64378.0         |      68619.0      |        1323377.5        |    971.7 
      B=32 L=256 D=16 none   |         2202.3         |       1587.4      |          31708.5        |    790.1 
      B=32 L=256 D=16 small  |        41631.1         |      41872.6      |         929014.4        |    848.1 
      B=32 L=256 D=16 large  |        32178.9         |      35337.5      |         651048.1        |    823.6 
      B=16 L=512 D=16 none   |         4464.3         |       3291.4      |          63352.0        |   1968.2 
      B=16 L=512 D=16 small  |        42085.8         |      43811.3      |         910916.2        |   2071.5 
      B=16 L=512 D=16 large  |        24021.5         |      24356.7      |         543484.5        |   1655.4 

Times are in microseconds (us).
```
<!-- /benchmark-soft -->

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
- `--only`: keep the cases whose label contains the substring at the start of a word, e.g. `--only "cuda n=16x16 s=32x32"`,
  to re-run only the cases that moved.
