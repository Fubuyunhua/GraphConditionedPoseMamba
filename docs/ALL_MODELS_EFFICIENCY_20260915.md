# All non-ablation model efficiency re-evaluation

Eight architecture configurations measured on RTX5090. GCPM ablations are excluded; identical detector/GT2D/fine-tuning architecture duplicates are counted once. W256D10 has no completed accuracy run: its row is architecture cost only.

## Protocol

- FP32, TF32 disabled, eager eval/inference_mode; no backward, optimizer or training.
- T243,J17; B1/B4; no flip and flip TTA. Real fixed first4 sorted H36M-SH test inputs. CPU loading/H2D is excluded.
- Random initialized architectures, not checkpoint accuracy replays. Actual weights do not change dense operation counts; no P1/P2 is newly measured.
- 10 warmups + 20 timed forwards x 3 seeded randomized repeats per case; synchronized host-wall and CUDA-event samples.
- A background GPU process exists. GPU/process snapshots are recorded per case. Timing is shared-device evidence, not an exclusive-device publication benchmark.
- Peak allocated memory is this benchmark process (model+input+intermediates), not whole-device nvidia-smi usage.

## Parameters and computational cost

| Model | Actual params | Vanilla THOP G (incomplete) | Contractions + scan G-MAC-equivalent | Per frame M |
|---|---:|---:|---:|---:|
| PoseMamba-A0 | 790,083 | 1.748 | 4.551 | 18.729 |
| GCPM-W64D8 | 800,083 | 2.342 | 4.602 | 18.940 |
| GCPM-W128D10 | 3,435,395 | 11.502 | 18.739 | 77.114 |
| GCPM-W128D20 | 6,836,355 | 22.999 | 37.474 | 154.215 |
| PoseMamba-L-file-architecture | 9,450,499 | 27.881 | 47.930 | 197.244 |
| GCPM-W160D32 | 16,534,531 | 57.288 | 89.413 | 367.956 |
| GCPM-W256D10 | 12,646,107 | 45.593 | 66.412 | 273.299 |
| GCPM-W256D16 | 20,192,451 | 72.942 | 106.255 | 437.263 |

Contraction counts agree exactly between corrected JIT and independent ATen dispatch. Scan-inclusive estimate is contraction_MACs +9BDLN/2+BDL per scan. Norm/activation/other pointwise arithmetic, exp/softplus and data movement are excluded; this is an analytic MAC-equivalent estimate, not hardware instruction counts. Vanilla THOP is shown separately because it omits unsupported functional/scan work and bare SSM parameters. True parameter sums, not THOP subtotals, are reported.

The PoseMamba-L-file row has2input channels; A0/GCPM use3(xy+confidence). Its9.450M actual parameters versus6.714M THOP subtotal explain the previous apparent discrepancy with the published6.7M, without proving exact checkpoint provenance.

## B1, flip TTA=False

| Model | Wall median ms | Wall P95 ms | CUDA median ms | Peak allocated MiB | Extra activation peak MiB | Successful repeats |
|---|---:|---:|---:|---:|---:|---:|
| PoseMamba-A0 | 5.959 | 6.071 | 5.938 | 69.6 | 57.4 | 3 |
| GCPM-W64D8 | 7.040 | 7.110 | 7.023 | 56.4 | 44.1 | 3 |
| GCPM-W128D10 | 10.651 | 10.741 | 10.632 | 107.9 | 85.5 | 3 |
| GCPM-W128D20 | 20.620 | 21.222 | 20.605 | 121.0 | 85.5 | 3 |
| PoseMamba-L-file-architecture | 23.530 | 24.023 | 23.515 | 153.8 | 108.6 | 3 |
| GCPM-W160D32 | 41.869 | 42.930 | 41.853 | 176.9 | 104.1 | 3 |
| GCPM-W256D10 | 18.925 | 18.999 | 18.905 | 223.9 | 166.4 | 3 |
| GCPM-W256D16 | 30.263 | 30.365 | 30.246 | 251.9 | 165.5 | 3 |

## B1, flip TTA=True

| Model | Wall median ms | Wall P95 ms | CUDA median ms | Peak allocated MiB | Extra activation peak MiB | Successful repeats |
|---|---:|---:|---:|---:|---:|---:|
| PoseMamba-A0 | 12.143 | 12.401 | 12.126 | 69.7 | 57.5 | 3 |
| GCPM-W64D8 | 14.309 | 14.575 | 14.293 | 56.5 | 44.2 | 3 |
| GCPM-W128D10 | 21.502 | 21.557 | 21.483 | 108.0 | 85.6 | 3 |
| GCPM-W128D20 | 41.324 | 42.471 | 41.310 | 121.1 | 85.6 | 3 |
| PoseMamba-L-file-architecture | 47.055 | 48.144 | 47.040 | 153.9 | 108.7 | 3 |
| GCPM-W160D32 | 83.899 | 85.638 | 83.884 | 176.9 | 104.2 | 3 |
| GCPM-W256D10 | 37.572 | 38.111 | 37.557 | 223.1 | 165.6 | 3 |
| GCPM-W256D16 | 60.012 | 60.798 | 59.998 | 252.0 | 165.6 | 3 |

## B4, flip TTA=False

| Model | Wall median ms | Wall P95 ms | CUDA median ms | Peak allocated MiB | Extra activation peak MiB | Successful repeats |
|---|---:|---:|---:|---:|---:|---:|
| PoseMamba-A0 | 12.232 | 12.298 | 12.214 | 236.1 | 223.8 | 3 |
| GCPM-W64D8 | 13.332 | 13.367 | 13.316 | 193.4 | 181.0 | 3 |
| GCPM-W128D10 | 30.125 | 30.317 | 30.110 | 370.4 | 347.9 | 3 |
| GCPM-W128D20 | 60.073 | 60.405 | 60.054 | 383.4 | 347.9 | 3 |
| PoseMamba-L-file-architecture | 71.848 | 72.356 | 71.832 | 475.0 | 429.7 | 3 |
| GCPM-W160D32 | 123.650 | 124.007 | 123.630 | 506.5 | 433.6 | 3 |
| GCPM-W256D10 | 63.901 | 64.004 | 63.885 | 743.6 | 686.0 | 3 |
| GCPM-W256D16 | 102.197 | 102.306 | 102.178 | 772.5 | 686.0 | 3 |

## B4, flip TTA=True

| Model | Wall median ms | Wall P95 ms | CUDA median ms | Peak allocated MiB | Extra activation peak MiB | Successful repeats |
|---|---:|---:|---:|---:|---:|---:|
| PoseMamba-A0 | 24.361 | 24.668 | 24.346 | 236.5 | 224.2 | 3 |
| GCPM-W64D8 | 26.790 | 26.855 | 26.773 | 194.7 | 182.2 | 3 |
| GCPM-W128D10 | 60.389 | 60.748 | 60.374 | 370.8 | 348.3 | 3 |
| GCPM-W128D20 | 120.187 | 120.884 | 120.173 | 383.8 | 348.3 | 3 |
| PoseMamba-L-file-architecture | 143.772 | 144.569 | 143.756 | 475.3 | 430.0 | 3 |
| GCPM-W160D32 | 247.446 | 248.126 | 247.427 | 506.8 | 434.0 | 3 |
| GCPM-W256D10 | 127.962 | 128.267 | 127.947 | 744.0 | 686.4 | 3 |
| GCPM-W256D16 | 204.542 | 204.995 | 204.521 | 772.9 | 686.4 | 3 |

## Evidence and limits

- Public machine-readable results: [efficiency summary](../evidence/efficiency_all_summary_20260915.json), covering all8 models and32 inference cases.
- Detailed counts.json and runtime.json are retained locally: all96 case-repeat records,60 latency samples per successful case, input hashes and GPU/process snapshots. Private machine paths/process metadata are not published.
- No accuracy/training results or old experiment logs were overwritten. No external process was stopped. The user authorized publication of results and methodology only; no model weights, data or profiling source scripts are included.
- For strict paper latency claims, repeat on an exclusive GPU with identical software/clock/power settings. Do not select the best timing sample or infer latency from MACs.

