# Results

| File | Content |
|---|---|
| [bench_summary.md](bench_summary.md) | `scripts/run_bench.sh` on the recommended configuration, and the comparison with the earlier PyTorch/XLA deployment |
| [accuracy_summary.md](accuracy_summary.md) | logits against FP32, FP8 against BF16 weights, audio word error rate, long context, GSM8K / MMLU |
| [bench/](bench/) | `summary.json` of each `scripts/run_bench.sh` run (`<profile>-<speculative option>.summary.json`) |

A `summary.json` holds, per `vllm bench serve` run, the `vllm bench serve` metrics (`bench`); per needle prompt the
times to first token and the answer (`prefix_cache`); per media input the times and answers of the three runs (`mm`);
the GSM8K / MMLU scores (`accuracy`); and the prefix-cache counters from `/metrics` (`metrics_delta`).
