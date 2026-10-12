# Lightning-Infer

**Lightning-Infer is a local inference engine for Qwen3.8-Flash-Next that runs one model across two
different NVIDIA GPUs in the same PC**, on Windows, so it runs as fast as a much smaller model:
**80-90 tokens/s** in conversation, **over 100** on coding work, with up to 512K tokens of context,
images and video.

Qwen3.8-Flash-Next is a mixture-of-experts model with 24,576 experts (about 70 GB at NVFP4): it does
not fit on any consumer GPU. Lightning-Infer splits it by layers across the two GPUs, keeps the
experts in system RAM, caches the ones the model is actually using in VRAM, and updates that cache as
the conversation changes topic.

## Why

- **Any two GPUs, even different ones.** A big card and a small one, on different links (even one in
  a slow chipset slot), with no NVLink and no P2P. Each card holds its own layers, its own share of the
  context and its own expert cache; only 20 KiB per token crosses between them.
- **No quality loss.** Output is bit-identical to running on a single GPU: same replies, same
  perplexity. Every optimization is checked this way before it goes in.
- **Built for coding agents.** MTP and n-gram drafts (code an agent re-reads and rewrites accepts
  long drafted runs), a prompt cache that reuses the conversation already read, robust tool calls.
- **Long prompts read fast.** 50K tokens of pasted code are read in about 21 seconds.
- **Images and video** in any request, with no cost to text speed.
- **Works with the clients you already use.** OpenAI APIs (Chat Completions and Responses) and
  Anthropic Messages: tested with Codex, opencode, pi and Claude Code-style requests.

## Performance

Measured on an RTX PRO 4000 Blackwell (24 GB, PCIe Gen3 x4) + RTX 5080 (16 GB, Gen4 x16),
Ryzen 9 5900X, 128 GB DDR4-3200, Windows 11. A 20-turn conversation with pasted code (~50K tokens),
average over the second half.

| | Decode (tok/s) | Peaks | Reading 50K tokens |
|---|---|---|---|
| **Lightning-Infer, conversation** | **80-90** | 120-145 | **~21 s** |
| **Lightning-Infer, coding agent** | **103-107** | | |
| Strata (Orca Q4_K_S) on the same two GPUs | 34 | 63 | 43-78 s |

With a 512K window filled in 50K-token steps: 69 tok/s at 63K, 62 at 257K, 55 at 514K.

## Requirements

- Two NVIDIA Blackwell GPUs (RTX 50 series / RTX PRO Blackwell), or just one.
- ~71 GB of free RAM for the experts (128 GB recommended).
- An NVMe SSD for the 52 GB n-gram volume.
- Windows 11 with a recent NVIDIA driver.
- The model: the Qwen3.8-Flash-Next NVFP4 Dense8 artifact (`.infernix`) and its `.ngram` volume.

## Build

You need Visual Studio 2022 Build Tools (MSVC 14.44), CUDA 13.3, CMake + Ninja and vcpkg in
`C:\vcpkg` (ffmpeg and curl, triplet `x64-windows`).

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
cmake -S . -B build-windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a ^
      -DINFERNIX_BUILD_APPS=ON
cmake --build build-windows --parallel 8
```

The programs land in `build-windows\bin`: `infernix-serve.exe` (server) and `infernix.exe` (command
line).

## Running

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
infernix-serve.exe <model>.infernix --ngram-volume <volume>.ngram ^
  --devices 0,1 --layer-split 28 --max-context 262144 ^
  --kv-dtype int8 --prefill-chunk 4096 --spec mtp --draft-tokens 4 --lm-head-draft ^
  --ngram-draft-tokens 7 --chat-template tools\chat_templates\qwen3_8.jinja ^
  --vision --tolerant-tool-calls --model-id qwen3.8-flash-next --host 127.0.0.1 --port 8891
```

- `--devices 0,1 --layer-split 28`: layers 0-27 on GPU 0, layers 28-47 and the head on GPU 1. Put
  more layers on the card with more VRAM; with `CUDA_DEVICE_ORDER=PCI_BUS_ID` the GPU numbers follow
  the slot order.
- `--device N` instead of `--devices` runs on a single GPU.
- 512K context: `--max-context 524288 --rope-yarn-factor 2`.
- `--chat-template` is needed for Codex (`developer` messages and `instructions`).

Then point your client at `http://127.0.0.1:8891/v1` with the model `qwen3.8-flash-next`.

## Documentation

- [docs/serving.md](docs/serving.md): server and APIs
- [docs/cli.md](docs/cli.md): command line
- [docs/qwen3_8-flash-next.md](docs/qwen3_8-flash-next.md): model guide and tuning

## Credits and license

Lightning-Infer is built on [Infernix](README.infernix.md), itself derived from
[NInfer](https://github.com/Neroued/ninfer) by Neroued. Licensed under Apache 2.0 ([LICENSE](LICENSE)).
Lightning-Infer's code was written with AI assistance (Claude Opus 5.5).
