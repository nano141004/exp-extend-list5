# Final Compiled Results

Compiled from `experiment-results/raw`.
All runs use the first 50 queries of each dataset.

Blank cells mean the raw result is not available yet.

## NDCG@10

| Dataset | Seq Top-40 | Seq Top-60 | Seq Top-80 | Seq Top-100 | Score-balanced Top-100 (seq cache / v1) | Score-balanced Top-100 (aligned cache / v2) |
|---|---:|---:|---:|---:|---:|---:|
| NFCorpus | 0.33718 | 0.34292 | 0.34212 | 0.34274 | 0.33845 | 0.33855 |
| SciFact | 0.72975 | 0.72829 | 0.73023 | 0.72683 | 0.71600 | 0.72262 |
| ArguAna | 0.46052 | 0.44007 | 0.41867 | 0.43436 | 0.40740 | 0.41691 |
| SCIDOCS | 0.16004 | 0.15644 | 0.17279 | 0.16678 | 0.16377 | 0.16164 |
| FiQA-2018 | 0.41760 | 0.43894 | 0.45958 | 0.45442 | 0.45830 | 0.45446 |

## Runtime: Seconds Per Query

| Dataset | Seq Top-40 | Seq Top-60 | Seq Top-80 | Seq Top-100 | Score-balanced Top-100 (seq cache / v1) | Score-balanced Top-100 (aligned cache / v2) |
|---|---:|---:|---:|---:|---:|---:|
| NFCorpus | 17.55392 | 16.48724 | 22.61349 | 24.87629 | 85.53880 | 40.89014 |
| SciFact | 23.47538 | 22.04464 | 30.52333 | 33.64508 | 119.90140 | 58.40244 |
| ArguAna | 22.67393 | 21.14882 | 29.60622 | 32.27358 | 118.74083 | 60.98355 |
| SCIDOCS | 22.26384 | 20.97219 | 29.04956 | 31.67722 | 118.70814 | 57.02675 |
| FiQA-2018 | 23.43233 | 22.30298 | 30.73836 | 33.72817 | 128.85962 | 56.70781 |

## Total Runtime: Seconds

| Dataset | Seq Top-40 | Seq Top-60 | Seq Top-80 | Seq Top-100 | Score-balanced Top-100 (seq cache / v1) | Score-balanced Top-100 (aligned cache / v2) |
|---|---:|---:|---:|---:|---:|---:|
| NFCorpus | 877.696 | 824.362 | 1130.675 | 1243.815 | 4276.940 | 2044.507 |
| SciFact | 1173.769 | 1102.232 | 1526.167 | 1682.254 | 5995.070 | 2920.122 |
| ArguAna | 1133.697 | 1057.441 | 1480.311 | 1613.679 | 5937.042 | 3049.178 |
| SCIDOCS | 1113.192 | 1048.609 | 1452.478 | 1583.861 | 5935.407 | 2851.337 |
| FiQA-2018 | 1171.617 | 1115.149 | 1536.918 | 1686.409 | 6442.981 | 2835.390 |

## Model Inference Calls: num_forward

`num_forward` is the number of ListT5 `model.generate(...)` calls made during the run.
It is a useful computational-cost proxy, especially when comparing the v1 and v2 cache behavior.

| Dataset | Seq Top-40 | Seq Top-60 | Seq Top-80 | Seq Top-100 | Score-balanced Top-100 (seq cache / v1) | Score-balanced Top-100 (aligned cache / v2) |
|---|---:|---:|---:|---:|---:|---:|
| NFCorpus | 2267 | 2011 | 2780 | 2969 | 11787 | 5609 |
| SciFact | 2955 | 2616 | 3646 | 3936 | 16628 | 7635 |
| ArguAna | 3003 | 2628 | 3717 | 3963 | 16563 | 7729 |
| SCIDOCS | 2970 | 2637 | 3655 | 3934 | 16628 | 7735 |
| FiQA-2018 | 2895 | 2591 | 3592 | 3865 | 16827 | 7535 |

## V2 Speed Check

| Dataset | v1 sec/query | v2 sec/query | Speedup (v1 / v2) |
|---|---:|---:|---:|
| NFCorpus | 85.53880 | 40.89014 | 2.09x |
| SciFact | 119.90140 | 58.40244 | 2.05x |
| ArguAna | 118.74083 | 60.98355 | 1.95x |
| SCIDOCS | 118.70814 | 57.02675 | 2.08x |
| FiQA-2018 | 128.85962 | 56.70781 | 2.27x |

## V2 Forward-Call Reduction

| Dataset | v1 num_forward | v2 num_forward | Reduction (v1 / v2) |
|---|---:|---:|---:|
| NFCorpus | 11787 | 5609 | 2.10x |
| SciFact | 16628 | 7635 | 2.18x |
| ArguAna | 16563 | 7729 | 2.14x |
| SCIDOCS | 16628 | 7735 | 2.15x |
| FiQA-2018 | 16827 | 7535 | 2.23x |
