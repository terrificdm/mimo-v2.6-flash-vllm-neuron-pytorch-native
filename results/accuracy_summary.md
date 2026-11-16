# Accuracy

There is no GPU on the instance. "Same as GPU" in this repository means either read from vLLM 0.29's source or
reproduced on the CPU with the GPU's dtype path (for example BF16 matrix multiplications with FP32 accumulation where
the GPU kernels do that).

## 1. Logits against an FP32 reference

16 prompts × 64 decode steps, teacher-forced, against a true FP32 reference of the full model. The comparison group is
a CPU emulation of the GPU algorithm with the same weights (the error a correct GPU deployment would have). Ratios
below 1 mean the device is closer to FP32 than the emulated GPU.

| Configuration | p95 ratio | p99 ratio | σ ratio |
|---|---|---|---|
| 8K, speculation off | 0.944 | 0.867 | 1.035 |
| 8K, MTP k=3 | 0.910 | 0.848 | 1.003 |
| 1M (expert placement), speculation off | 0.841 | 0.861 | 0.997 |
| 1M (expert placement), MTP k=3 | 0.854 | 0.842 | 1.004 |

## 2. FP8 against BF16 weights

Same server, FP8 weights (the deployment) against the checkpoint dequantized to BF16, with the lm-eval harness:

| Task | BF16 | FP8 | Paired difference |
|---|---|---|---|
| GSM8K (1,319 questions), greedy | 0.9712 | 0.9712 | 0.0000 |
| GSM8K, sampled | 0.9704 | 0.9719 | +0.0015 |
| MMLU-Pro (1,000 questions), greedy | 0.779 | 0.778 | −0.0010 |
| MMLU-Pro, sampled | 0.781 | 0.777 | −0.0040 |

All differences are within one paired standard error. MMLU-Pro with `max_tokens` 4096 truncates 9-11 % of the
answers on both sides, so the absolute scores are not comparable with published ones. On the final configuration,
MMLU-Pro with GPU-aligned sampling and `max_tokens` 8192 gave 0.797 and 0.804 in two runs.

## 3. Audio

Word error rate on 1,551 words of LibriSpeech test-clean (1M profile):

| Path | WER |
|---|---|
| device, speculation off | 2.19 % / 2.26 % (two runs) |
| device, MTP k=3 | 2.32-2.39 % |
| audio embeddings computed on the CPU with the GPU dtype path, injected into the device's encoder cache | 1.93 % |
| FP32 Hugging Face audio embeddings injected into the device's encoder cache | 2.19 % |

The criterion was device ≤ GPU path + 0.3 percentage points without speculation. The extra errors with MTP come from
near-tied tokens on the text side (the FP32-embedding run makes the same number of errors as the device, so the
audio towers are not the cause); ASR evaluation should use `--spec off`.

## 4. Long context

- Needle in a haystack at 32K and 128K tokens: 20/20 with FP8 and with BF16.
- 1M profile: a 999,996-token needle prompt at 50 % depth answered correctly (first token after 157 s).

## 5. `scripts/run_bench.sh` (GSM8K / MMLU, 200 questions each)

0-shot chat, thinking off, greedy, 4 workers. Questions: [data/task_eval/](../data/task_eval/).

| Configuration | GSM8K | MMLU |
|---|---|---|
| `1m`, MTP k=3 | 98.0 % | 79.5 % |
| `1m_c4`, speculation off | 97.0 % | 79.5 % |
| `1m_c4`, MTP k=3 | 98.0 % | 79.5 % |
| earlier PyTorch/XLA deployment (for reference) | 98.0 % | 80.5 % |

With 200 questions one question is 0.5 points. Paired per question, `1m_c4` without speculation and the XLA
deployment differ on 4 GSM8K and 4 MMLU questions (exact McNemar p ≈ 0.6 for both): no significant difference.
