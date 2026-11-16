# 教程：用 vLLM Neuron（PyTorch native 后端）部署 MiMo-V2.6-Flash-MOPD

[English](README.md)

本教程在一台 AWS **trn2.48xlarge** 上，用 vLLM 和 vllm-neuron 插件的 **PyTorch native 后端**部署
[XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)：同一个 server 提供 1M token
上下文、图片 / 视频 / 音频输入、前缀缓存和 MTP 投机解码。模型支持是 [patches/](patches/) 里打在 vllm-neuron 0.29 上的
补丁；直接使用原始 checkpoint，不需要转换权重。

流程：启动实例 → 建 venv 并安装打过补丁的 vllm-neuron → 下载权重 → 启动 server → 发请求 → 跑 benchmark。

## 目录

- [概览](#概览)
- [前提条件](#前提条件)
- [第 1 步：安装软件](#第-1-步安装软件)
- [第 2 步：下载权重](#第-2-步下载权重)
- [第 3 步：启动 server](#第-3-步启动-server)
- [第 4 步：发请求](#第-4-步发请求)
- [第 5 步：冒烟测试](#第-5-步冒烟测试)
- [第 6 步：Benchmark](#第-6-步benchmark)
- [其他配置](#其他配置)
- [精度](#精度)
- [限制](#限制)
- [故障排查](#故障排查)
- [相关资源](#相关资源)
- [许可证](#许可证)

## 概览

推荐配置（`1m` profile，MTP k=3）提供的功能，都用本仓库的代码验证过：

- **1M 上下文**（1,048,576 token），FP8 权重（块缩放 FP8 线性层；专家由 checkpoint 的 MXFP4 无损编码成 FP8）加 BF16
  KV cache，与官方 H100 部署的精度方案相同。
- **图片、视频、音频**在同一个 server 里；视觉编码器和两个音频塔都跑在 Trainium 上。
- **前缀缓存**：重复或续写的长 prompt 几秒就出首 token。
- **MTP 投机解码**，用 checkpoint 自带的 3 个 MTP 层；也可以选 DFlash 或关闭投机。
- 最多 **8 个并发请求**；vLLM 的 `mimo` reasoning / tool parser；结构化输出；采样参数与 GPU 上的 vLLM 语义相同。

关键数字（一台 trn2.48xlarge，`1m` profile，MTP k=3，`scripts/run_bench.sh`；详见[第 6 步](#第-6-步benchmark)）：

| 测试 | 结果 |
|---|---|
| 短请求（输入 512 / 输出 128），并发 1 | TTFT 214 ms，TPOT 15.0 ms |
| 输入 8,192 / 输出 512，并发 1 | TTFT 890 ms，TPOT 10.7 ms |
| 940K / 30K token 的首 token（第一次请求） | 148.8 s / 3.24 s |
| 同一 prompt 再发一次（前缀缓存命中），940K / 30K | 3.38 s / 0.30 s |
| GSM8K / MMLU（各 200 题，0-shot，greedy） | 98.0 % / 79.5 % |
| 编译缓存为空时第一次启动 | 31 分钟 |

## 前提条件

**实例**

- 一台 **trn2.48xlarge**（16 个 Trainium2 设备，LNC=2 下 64 个逻辑 NeuronCore，每个逻辑核 24 GiB HBM，主机内存 2 TB），
  可以用 EC2 Capacity Block for ML 或按需容量。
- 安全组只放行入站 **SSH**。server 监听 `127.0.0.1:8000`，用 `ssh -L 8000:127.0.0.1:8000 ...` 访问。
- 根盘至少 512 GB（DLAMI 对 Trn 实例的最低要求）；权重、venv 和编译缓存放在本地 NVMe 上（见下文）。

**AMI 与软件**：Neuron SDK **2.33.0** 的 Neuron multi-framework DLAMI（Ubuntu 24.04，"Deep Learning AMI Neuron
(Ubuntu 24.04)"；所在 Region 的 AMI ID 通过
[Neuron multi-framework DLAMI 指南（2.33.0）](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/setup/multiframework-dlami.html)
里的 SSM 参数查询）。它自带 venv `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`；本教程按它的包清单另建一个
venv，不修改 DLAMI 自带的 venv。

> [!IMPORTANT]
> 只在下面这些版本上验证过：
>
> | 组件 | 版本 |
> |---|---|
> | Neuron SDK | 2.33.0 |
> | vllm-neuron | 0.29.0.1.2.0 + 本仓库的补丁 |
> | vLLM | 0.29.0 |
> | torch | 2.13.0 |
> | neuronx-cc | 2.28.7631.0 |
> | nki | 0.7.0 |
> | 后端 | vllm-neuron PyTorch native（`VLLM_NEURON_BACKEND=neuron_native`） |
>
> 不要在服务 venv 里升级或添加包。

**磁盘**：权重 178 GB + 服务 venv 约 13 GB + 每个启动过的配置约 25 GB 编译缓存（`1m` 和 `8k` 的所有投机选项都启动过约
150–200 GB）。把 4 块实例存储 NVMe 组成 RAID 0（约 6.9 TiB；先用 `lsblk` 确认设备名）：

```bash
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1
sudo mkfs.xfs /dev/md0
sudo mkdir -p /mnt/nvme && sudo mount -o noatime /dev/md0 /mnt/nvme && sudo chown "$USER:$USER" /mnt/nvme
mkdir -p /mnt/nvme/mimo
```

> [!WARNING]
> 实例停止后实例存储上的数据会丢失。重启后执行：
> `sudo mdadm --assemble --scan && sudo mount -o noatime /dev/md0 /mnt/nvme`。

所有脚本把文件放在 `MIMO_ROOT` 下（默认 `/mnt/nvme/mimo`）；要换目录就先 export 它（必须是本地盘）。路径和默认值见
[scripts/env.sh](scripts/env.sh)。

## 第 1 步：安装软件

约 8 分钟。

```bash
cd /mnt/nvme/mimo
git clone https://github.com/terrificdm/mimo-v2.6-flash-vllm-neuron-pytorch-native.git
cd mimo-v2.6-flash-vllm-neuron-pytorch-native
bash scripts/check_host.sh      # 只读检查：实例类型、DLAMI venv、16 个空闲设备、内存、磁盘
bash scripts/setup_venv.sh      # 新建 /mnt/nvme/mimo/venv：DLAMI vLLM venv 的全部包 + av、soundfile
bash scripts/setup_source.sh    # clone vllm-neuron、打补丁、核对哈希、editable 安装
```

- `check_host.sh` 最后一行是 `all checks passed`。
- `setup_venv.sh` 用 `--no-deps` 从 PyPI 和 Neuron pip 源安装 DLAMI venv 的完整包清单
  （[requirements/serve-freeze.txt](requirements/serve-freeze.txt)），再逐行核对结果（约 7 分钟）。
- `setup_source.sh` 把 [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron) clone 到
  `release-0.29.0.1.2.0`，用 `git am` 打上 3 个补丁，按 [sources.lock.json](sources.lock.json) 核对树哈希，然后
  editable 安装（必须这样装：vLLM 通过 entry point 找到插件）。最后一行是 `vllm-neuron installed (editable)`。

## 第 2 步：下载权重

```bash
tmux new -d -s download "bash scripts/download_weights.sh > /mnt/nvme/mimo/download.log 2>&1"
tail -f /mnt/nvme/mimo/download.log          # 最后是 "weights ready: ..."；失败时退出码 3
```

178 GB、154 个文件，revision `b12f5573350bb1704513fcf119fca48e498b0423`，下载到 `/mnt/nvme/mimo/hf`。权重是公开的，不需要
token。下载完脚本按 [weights.lock.json](weights.lock.json) 核对每个文件的大小和 sha256。中断后重新执行同一条命令即可续传。
checkpoint 必须带 `audio_tokenizer/`（音频塔）和 `dflash/`（DFlash 草稿），脚本会检查。

## 第 3 步：启动 server

先为 server 开一个 tmux 会话（Ctrl-b d 退出，`tmux attach -t serve` 回来）：

```bash
tmux new -s serve
```

在会话里、仓库目录下执行：

```bash
bash scripts/serve.sh            # profile 1m，MTP k=3
```

在另一个 shell 里等它就绪：

```bash
bash scripts/wait_ready.sh       # 每 5 分钟打印一次最新日志；最后是 "ready ... ['mimo-v2.6-flash']"
```

**耗时**：编译缓存为空时，第一次启动约 **31 分钟**：约 10 分钟把权重加载到 64 个核上，然后并行编译各个图
（每个 prefill 档位、每个 decode 档位一个，外加编码器和草稿层）。之后同一配置的启动从缓存加载，约 14 分钟。日志是
`/mnt/nvme/mimo/logs/serve-latest.log`，里面有 `Using neuron_native backend`。

`scripts/serve.sh` 从 [config/serve_args.json](config/serve_args.json) 渲染出命令，`--dry-run` 只打印不启动。`1m`
profile 加 MTP k=3 的命令是：

```bash
env VLLM_NEURON_BACKEND=neuron_native NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_SWITCH_CC=1 \
  TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=43200 NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=12 VLLM_RPC_TIMEOUT=1800000 \
  vllm serve <checkpoint> \
  --trust-remote-code --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  --generation-config vllm --tensor-parallel-size 64 --enable-expert-parallel \
  --max-model-len 1048576 --max-num-batched-tokens 8192 --max-num-seqs 8 --num-gpu-blocks-override 38587 \
  --enable-prefix-caching --prefix-cache-retention-interval 1024 \
  --limit-mm-per-prompt '{"image": 64, "video": 4, "audio": 8}' --allowed-local-media-path <repo>/assets \
  --served-model-name mimo-v2.6-flash --host 127.0.0.1 --port 8000 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --additional-config '{"neuron_config": {"quantization": "fp8", "num_batched_tokens_buckets": [512, 1024, 8192],
      "num_seqs_buckets": [1, 4, 8], "on_device_sampling_config": {"all_greedy": false},
      "mimo_attn_kernel": "nki", "mimo_expert_placement": "<repo>/config/placement3_s5.json"}}'
```

请用 `scripts/serve.sh` 启动，不要手敲这条命令：脚本还会给每个配置设单独的编译缓存目录
（[原因](docs/configuration.md#5-compile-cache)），并把缓存、临时文件和日志放到 `MIMO_ROOT` 下。

| 参数 | 作用 |
|---|---|
| `VLLM_NEURON_BACKEND=neuron_native` | vllm-neuron 的 PyTorch native 后端 |
| `--tensor-parallel-size 64 --enable-expert-parallel` | 用满 64 个逻辑核；256 个专家分到 64 个 rank |
| `--max-model-len 1048576`、`--max-num-batched-tokens 8192` | 1M 上下文；更长的 prompt 按 8192 token 一段做 prefill |
| `num_batched_tokens_buckets [512, 1024, 8192]` | prefill 图的档位；每一步补齐到能装下它的最小档位 |
| `--max-num-seqs 8`、`num_seqs_buckets [1, 4, 8]` | 最多 8 个并发请求；1、4、8 行三张 decode 图 |
| `--num-gpu-blocks-override 38587` | KV 池 38,587 × 32 token；必须设，否则图放不进 HBM |
| `--enable-prefix-caching --prefix-cache-retention-interval 1024` | 前缀缓存，包括 SWA 层上"在 prompt 末尾续写"的命中 |
| `quantization fp8` | 从原始 checkpoint 读出 FP8 线性层和 FP8 专家 |
| `mimo_attn_kernel nki` | 模型自带的 NKI 注意力 kernel（prefill 和 decode） |
| `mimo_expert_placement` | 静态专家放置，给热专家加 prefill 副本（[说明](docs/configuration.md#3-expert-placement-configplacement3_s5json)） |
| `--speculative-config` mtp，3 个 token | 用 3 个 MTP 层做投机解码（[选项](docs/speculative-decoding.md)） |
| `--limit-mm-per-prompt` | 每个请求最多 64 张图、4 段视频、8 段音频 |
| `--generation-config vllm` | 模型的 generation_config.json 不覆盖请求的默认值 |
| `NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=12`、`TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=43200` | 让第一次编译不超出主机内存 |

停止 server：在它的 tmux 窗口里按 Ctrl-C，或执行 `bash scripts/stop.sh`。

> [!CAUTION]
> server **没有鉴权**。保持 `--host 127.0.0.1`；需要让别的机器访问时，在前面加一个带鉴权的代理。

## 第 4 步：发请求

模型名 `mimo-v2.6-flash`；默认开启思考（先流式返回 `reasoning`，再返回 `content`）。

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95,
  "messages": [{"role": "user", "content": "用三句话介绍一下长城。"}]}'
```

关闭思考：加 `"chat_template_kwargs": {"enable_thinking": false}`。流式：加 `"stream": true`。

用 `file://` URL 发图片 / 视频 / 音频（文件必须在 `--allowed-local-media-path` 下，默认是仓库的 `assets/`；要换目录，在
`scripts/serve.sh` 之前设 `MEDIA_DIR`）：

```bash
A=$PWD/assets
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "file://'"$A"'/number_739.png"}},
    {"type": "text", "text": "What number is written in the image? Answer with the number only."}]}]}'
# 视频：{"type": "video_url", "video_url": {"url": "file://'"$A"'/digits_1234.mp4"}}   -> "1, 2, 3, 4"
# 音频：{"type": "audio_url", "audio_url": {"url": "file://'"$A"'/audio/jfk.wav"}}     -> 肯尼迪演讲
```

工具调用（OpenAI 格式，`"tools": [...]`、`"tool_choice": "auto"`）、结构化输出（`response_format` 加 JSON schema）和
OpenAI Python SDK 的用法与普通 vLLM server 相同。

[scripts/api_tests/ask.py](scripts/api_tests/ask.py) 是一个小的流式客户端，把本地文件转成 base64 发送（只用标准库）：

```bash
python3 scripts/api_tests/ask.py "17×23 等于多少？"
python3 scripts/api_tests/ask.py --no-think --image assets/number_739.png "图里写的是什么数字？"
python3 scripts/api_tests/ask.py --no-think --video assets/digits_1234.mp4 "视频里依次出现了哪些数字？"
python3 scripts/api_tests/ask.py --no-think --audio assets/audio/jfk.wav "把这段音频原样转写出来。"
```

它打印 `思考`、`回答`、`[结束原因]`、`[用量]` 和 `[首字时间]`。

## 第 5 步：冒烟测试

```bash
bash scripts/smoke_test.sh       # 8k profile 上用：bash scripts/smoke_test.sh --no-mm
```

13 项检查：模型列表、英文 / 中文 / 算术对话、completions、流式、思考、工具调用、JSON schema、4 个并发请求、图片、视频、
音频。预期输出 `smoke: 13 pass, 0 fail`。结果在 `/mnt/nvme/mimo/results/smoke.{json,md}`。

## 第 6 步：Benchmark

```bash
OUT=/mnt/nvme/mimo/bench/1m-mtp3 bash scripts/run_bench.sh       # 约 15 分钟；SMOKE=1 跑缩小版
```

[scripts/run_bench.sh](scripts/run_bench.sh) 只通过 HTTP 测试：`vllm bench serve` 随机 prompt（512 / 8K / 32K / 128K
token，并发 1 和 4）、needle 长 prompt（30K 和 940K token：第一次请求、完全相同的重复、追问一轮）、图片 / 视频 / 音频请求，
以及 GSM8K / MMLU（各 200 题）。汇总在 `$OUT/summary/summary.md`。

推荐配置（`1m`，MTP k=3）的结果：

| 负载 | TTFT | TPOT | 输出 tok/s |
|---|---|---|---|
| 输入 512 / 输出 128，并发 1 | 214 ms | 15.0 ms | 60.4 |
| 输入 512 / 输出 128，并发 4 | 396 ms（平均） | 33.9 ms | 101.2 |
| 输入 8,192 / 输出 512，并发 1 | 890 ms | 10.7 ms | 80.9 |
| 输入 8,192 / 输出 512，并发 4 | 1,733 ms（平均） | 29.6 ms | 115.5 |
| 输入 32,768 / 输出 256，并发 1 | 3,687 ms | 16.2 ms | 32.8 |
| 输入 131,072 / 输出 128，并发 1 | 15,762 ms | 17.1 ms | 7.1 |
| 30K needle prompt：第一次 / 重复 / 追问 | 3.24 / 0.30 / 0.31 s | — | — |
| 940K needle prompt：第一次 / 重复 / 追问 | 148.8 / 3.38 / 3.20 s | — | — |
| 图片 / 视频 / 音频（TTFT，3 次中位数） | 0.22 / 0.33 / 0.22 s | — | — |

GSM8K 98.0 %、MMLU 79.5 %（各 200 题）。每个逻辑核的 HBM：就绪时 20.96 GiB，940K prompt 之后最多 21.67 GiB（上限
24 GiB）。编译缓存为空时第一次启动：1,877 s。

同一套测试也在这个模型之前的 PyTorch/XLA 后端部署（Neuron SDK 2.32、vllm-neuron 0.24）上跑过；对比和 1M 4 并发配置的
结果见 [results/bench_summary.md](results/bench_summary.md)。

说明：

- `vllm bench serve` 的随机 token prompt 是投机解码最差的情况（草稿很少被接受）；在自然文本上，MTP k=3 的输出吞吐是关投机的
  1.4–2.0 倍（[speculative-decoding.md](docs/speculative-decoding.md#2-how-much-it-helps)）。
- 一个请求可以占满整个 1M KV 池；同时有多个长请求时，后面的要排队。

## 其他配置

[config/serve_args.json](config/serve_args.json) 有 4 个 profile 和 5 个投机选项：

| Profile | 上下文 | 并发 | 多模态 | 说明 |
|---|---|---|---|---|
| `1m`（默认） | 1M | 8 | 有 | 推荐 |
| `1m_c4` | 1M | 4 | 有 | 并发 ≤ 4 时延迟更低 |
| `8k` | 8K | 32 | 无 | 大量短文本请求 |
| `8k_torch` | 8K | 32 | 无 | 用 torch 注意力代替 NKI kernel |

```bash
bash scripts/serve.sh --profile 8k                  # --spec off | mtp1 | mtp2 | mtp3 | dflash
```

评测、回归测试、数据生成需要 greedy 输出逐位一致或需要 `seed` 时，以及 ASR 评测，请用 `--spec off`。每个新的（profile，
选项）第一次启动时都要编译自己的图。详见 [docs/configuration.md](docs/configuration.md)、
[docs/speculative-decoding.md](docs/speculative-decoding.md)。

## 精度

- **Logits**：对真 FP32 参考（16 条 prompt × 64 步），4 个默认配置的误差都在 GPU 算法 CPU 仿真的误差以内（p95 比
  0.84–0.94，σ 比 1.00–1.04）。
- **GSM8K / MMLU-Pro**（1,319 / 1,000 题）：FP8 与 BF16 权重的差在 1 个配对标准误以内（GSM8K greedy 都是 0.9712）。
- **音频**：关投机时 LibriSpeech test-clean 1,551 词的词错率 2.19–2.26 %。
- `scripts/run_bench.sh`：GSM8K 98.0 %，MMLU 79.5 %（各 200 题）。

方法和完整数字：[results/accuracy_summary.md](results/accuracy_summary.md)、
[docs/design-notes.md](docs/design-notes.md#accuracy)。

## 限制

详见 [docs/limitations.md](docs/limitations.md)。

- 只在 Neuron SDK 2.33.0 / vllm-neuron 0.29.0.1.2.0 的 PyTorch native 后端上验证过。
- 开投机解码时带 `seed` 的请求返回 HTTP 400；投机解码与 `--spec off` 的输出不逐位一致。
- 1M profile 上单个图片、视频或音频条目最多 114,688 token（约 128K token 及以上的视频不支持）；默认 32 帧抽样的 4K 视频
  超限。
- `1m` 最多同时运行 8 个请求；一个 1M token 的请求就占满 KV 池。
- 1M token 的第一次请求约 2.5–3 分钟出首 token（前缀缓存命中只要几秒）。
- 第一次编译约 31 分钟；权重和编译缓存在实例存储上。

## 故障排查

更多情况见 [docs/troubleshooting.md](docs/troubleshooting.md)。

- **20 分钟后还没就绪**：第一次启动在编译，用 `scripts/wait_ready.sh` 看进度。
- **`fp8 quantization is currently not supported in neuron`**：执行 `scripts/setup_source.sh`（editable 安装）。
- **带 `seed` 的请求返回 400**：开着投机解码；用 `--spec off`。
- **400 `... encoder-cache blocks ...`**：单个媒体条目太大；减少抽帧或降低分辨率。
- **只有思考、没有回答**：调大 `max_tokens` 或关闭思考。
- **重启后 `/mnt/nvme` 是空的**：重新组装 RAID（实例停止再启动后数据已经没了）。

## 相关资源

- [Hugging Face 上的 XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
- [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
- [Neuron multi-framework DLAMI（2.33.0）](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/setup/multiframework-dlami.html)
- [mimo-v2.6-flash-vllm-neuron-pytorch-xla](https://github.com/terrificdm/mimo-v2.6-flash-vllm-neuron-pytorch-xla)：
  同一模型在 vllm-neuron PyTorch/XLA 后端（Neuron SDK 2.32）上的部署
- 本仓库：[配置](docs/configuration.md)、[投机解码](docs/speculative-decoding.md)、[设计说明](docs/design-notes.md)、
  [限制](docs/limitations.md)、[故障排查](docs/troubleshooting.md)、[开发](docs/development.md)、
  [结果](results/README.md)、[发布前检查](docs/release-checklist.md)。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。模型权重由小米以 MIT 许可证发布，不包含在本仓库里。第三方来源列在 [NOTICE](NOTICE)。
