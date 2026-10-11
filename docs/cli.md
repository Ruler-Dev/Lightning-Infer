# Infernix CLI

`build/apps/infernix` runs one request against one v3 `.infernix` (or `.ninfer`) artifact. Build Infernix and
download an artifact using the [project README](../README.md) before following this guide.

The examples use Qwen3.8-27B NVFP4 with FP8 KV storage.

## Text input

```bash
./build/apps/infernix models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one of `--prompt` and `--messages` is required. The CLI normally omits `--kv-capacity`, so
the shared Main Text KV pool follows the example's 32,768-token `--max-context`.

Answer content is streamed to stdout. Human-readable startup milestones and runtime errors are
written to stderr without service timestamps. Reasoning and the CLI result report (timings,
throughput, GPU memory, token IDs when requested, and speculative-decoding statistics) also use
stderr as unprefixed product output, so stdout can be redirected independently. On a terminal,
weight materialization is one transient progress line followed by a compact Engine-ready summary.
Redirected stderr contains persistent readable progress for long loads and no carriage returns or
ANSI escapes. `--log-level debug` exposes every startup phase. FFmpeg's media-decoding messages are
records prefixed `media |`: FFmpeg errors are warnings and everything milder is `debug`. Option and
local prompt/message input failures remain direct command diagnostics:

```bash
./build/apps/infernix models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype fp8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` overrides the artifact's built-in template with a local Jinja file.
Changes to the file take effect after restarting Infernix:

```bash
./build/apps/infernix models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

`--grammar-file FILE` constrains the answer with a GBNF grammar whose entry rule is `root`:

```bash
printf 'root ::= "yes" | "no"\n' > answer.gbnf
./build/apps/infernix models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Is 17 prime?" --no-thinking --grammar-file answer.gbnf --max-new 64
```

`--choice TEXT` selects a literal candidate; repeat the flag to supply the candidate set.
`--regex PATTERN` constrains the complete answer to a regular expression:

```bash
./build/apps/infernix models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Classify this review: the service was excellent." --no-thinking \
  --choice positive --choice neutral --choice negative

./build/apps/infernix models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return ticket TASK-0042." --no-thinking --regex '(BUG|TASK)-[0-9]{4}'
```

Candidates preserve exact text; `--choice ''` explicitly permits empty content. `--regex ''`
permits only empty content. See the [language contract](maintainer/constrained-decoding.md#41-gbnf--regex--choice)
for supported regex syntax.

`--json-object` constrains the answer to a JSON object. `--json-schema-file FILE` applies a JSON
Schema. All output constraint options are mutually exclusive:

```bash
printf '%s\n' '{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}' > answer.schema.json
./build/apps/infernix models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return the answer to 6 times 7 as JSON." --no-thinking \
  --json-schema-file answer.schema.json --max-new 64
```

GBNF supports recursive rules, Unicode character classes and repetition. All constraint modes work with
ordinary decoding, MTP, DFlash and DFlash2, and on Qwen3.8-Flash-Next with its MTP and n-gram
copy rounds. Thinking may precede the constrained answer; an output
limit or cancellation can leave it incomplete. JSON modes can accompany tools supplied in messages:
the answer is either JSON or a tool-call sequence. GBNF, choice and regex require no active tools.
Constraints reject custom stops and `--raw-output`. JSON uses compact separators and declared property order. Describe
the desired content in the prompt; the schema is not added to it automatically. See the
[supported schema subset](maintainer/constrained-decoding.md#42-the-json-and-schema-execution-contract).

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or
`--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with
`--no-thinking`. The template interprets the selected effort; a value the template rejects renders
as the nearest one it accepts (see [serving](serving.md)). `--greedy` selects exact argmax
decoding independently.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking`, but it can be
combined with `--reasoning-effort`.

`--reasoning-loop off|stop|conclude` (default `off`) ends the reply (`stop`) or closes the thinking
with the same control span and lets the model answer (`conclude`) when the open thinking keeps
repeating whole passages: every 512 thinking tokens, 25 % of the last 2,000 words inside a 12-word
passage seen three times. The check and its limits are described in
[serving](serving.md), under `--reasoning-loop`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
When the effective output capacity extends beyond the thinking budget, it must have room for the
complete tokenizer-derived control suffix plus one post-close model token; an undersized request is
rejected rather than truncating the suffix. Normal output sends the inserted guidance to stderr as
reasoning. `--print-token-ids` includes the inserted IDs, while `--raw-output` preserves the raw
control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/apps/infernix models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash/DFlash2 weights and state and the optimized proposal head;
- `--spec mtp`, `--spec dflash` (35B-A3B), and `--spec dflash2` (Qwen3.8-27B) load only
  the selected speculative backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input.
- the one-request CLI disables cross-request history and Host context backing, so it does not
  reserve an extra Device checkpoint StateImage or retain continuations.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine
started without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may
be enabled together; these backends apply to generated-text decode after multimodal prefill and does not
accelerate Vision encode. The default speculative and Vision settings produce the smallest resident
profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/infernix models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype fp8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
The selected template formats these roles. The maintained Qwen templates keep system/developer
messages at their input positions.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP with one to five draft positions, or the
35B-A3B DFlash or Qwen3.8-27B DFlash2 backend with one to fifteen. Both masked-draft backends
may be combined with `--vision`.
`--lm-head-draft` selects the optimized proposal head and requires a selected backend:

```bash
./build/apps/infernix models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/infernix models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype fp8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select
`--spec dflash2 --draft-tokens 7`, optionally with `--lm-head-draft` and `--vision`.
DFlash2 accepts every draft count from 1 through 15; seven is the checkpoint recommendation.
Both `groupwise-int` and `nvfp4` artifacts use the same Engine route, including CUDA Graph,
concurrent requests, sampling penalties, and prefix reuse. An artifact without the companion
weights reports a missing DFlash2 component when selected. Vision, MTP and DFlash follow the same
rule: their weights are required only when that component is enabled at startup.

Only one speculative backend can be enabled per Engine. The published [performance results](performance.md)
use MTP with three draft tokens and DFlash with seven draft tokens (block length eight), both with
the optimized proposal head. DFlash accepts one to fifteen draft tokens; seven forms the measured
block length eight, while fifteen uses the maximum supported block length sixteen.

## Common options

The table lists executable defaults. The examples above select FP8 KV and MTP3.

| Option | Meaning | Default |
|---|---|---:|
| `--rope-yarn-factor F` | startup-fixed runtime YaRN factor, finite `[1,4]`; extends allowed ceiling only | `1` |
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--vram-headroom-mib N\|auto` | VRAM in MiB that startup sizing leaves free. Qwen3.5: after `--kv-capacity auto` sizes the KV pool (requires `auto`; `auto` = 1024). Qwen3.8-Flash-Next: after the expert cache takes the rest, for the display and other programs (`auto` = 1024 with a display on the GPU, 256 without; [details](qwen3_8-flash-next.md#vram)) | `auto` |
| `--vram-past-budget` | Qwen3.8-Flash-Next on Windows: also size the expert cache into the VRAM the Windows per-process budget withholds (~0.6 GiB, ~240 more expert frames on an RTX 5090), measured at startup; that memory may be moved to system memory if another program needs VRAM ([details](qwen3_8-flash-next.md#vram)) | off |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128; rounded down to whole prompt-attention waves (896 tokens for the 24-head model on RTX 5090) with INT8, NVFP4, FP8, K8V4, VQ2 or K4V2 KV unless an original prompt kernel is selected | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device ordinal, in PCI bus order ([device order](#device-order)) | `0` |
| `--devices A,B` | Qwen3.8-Flash-Next layer split across two GPUs: the first layers on `A`, which also runs the engine, the remaining layers, the output head and the MTP drafter on `B`; `A,A` splits on one GPU (a test mode). One ordinal is the same as `--device N`; the two flags cannot be combined ([layer split](#layer-split)) | one device |
| `--layer-split K\|auto\|@FILE` | where the split falls: after layer `K` (at least 1 and below the layer count), chosen at startup, or read from a plan file; requires two devices | `auto` with two devices |
| `--layer-split-report` | report the split planner's table of candidate `K` at startup; requires two devices | off |
| `--split-handoff auto\|event\|gate` | how decode and verification rounds pass the hidden state between the GPUs: `event` copies it through pinned memory behind a cross-device event wait; `gate` publishes it with a word that a bounded kernel on the next GPU polls (no host step between the GPUs, but that kernel spins); `auto` is `event`; requires two devices, and `gate` two different GPUs | `auto` |
| `--handoff-timeout-ms N` | bound of a `gate` handoff's wait, `1..1500`; a wait that runs out fails the round with a recoverable error instead of reaching Windows' 2 s GPU timeout; requires two devices | `1000` |
| `--ram-headroom-mib N` | physical Host RAM in MiB that pinned weights and caches leave free for the system; startup refuses a lock that would take it (Qwen3.8-Flash-Next plans every allocation against it in its RAM ledger) | `2048` |
| `--expert-ram-mib N\|auto` | Qwen3.8-Flash-Next: Host RAM in MiB for the routed experts; less than all of them need (~64.5 GiB) keeps that many in RAM and reads the rest from the artifact ([SSD expert tier](qwen3_8-flash-next.md#ssd-expert-tier)); more than is free is an error | `auto` (what the ledger leaves) |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4\|vq2\|k4v2` | KV-cache storage | `bf16` |
| `--use-original-int8-prefill-kernel` | prefill INT8 KV with the original prompt-attention kernel; requires `--kv-dtype int8` | fast kernel |
| `--prefill-8bit-pv` | force prompt attention's P×V onto 8-bit Tensor Cores where its kernel has both forms (the fast INT8 and NVFP4 prompt kernels, K8V4's tiled kernel and the VQ2 and K4V2 prompt kernel); INT8 and K4V2 only override their default, FP16 P×V, which is 5-7 % slower on long prompts but has half (INT8) or 1.09x (K4V2) the KL divergence from a BF16 KV reference | INT8, K4V2 |
| `--no-prefill-8bit-pv` | force FP16 P×V for NVFP4, K8V4 and VQ2 KV too, without their default 8-bit forms (5-7 % more end-to-end long-prompt prefill time; those forms stay within 1.1x FP16's KL divergence from BF16 KV) | NVFP4, K8V4, VQ2 |
| `--use-original-nvfp4-prefill-kernel` | prefill NVFP4 KV with the tiled prompt-attention kernel; requires `--kv-dtype nvfp4` | fast kernel over more than 768 visible keys |
| `--prefill-split-workspace-mib N` | workspace in MiB, `0..16384`, for splitting one prompt-attention launch's keys across SMs (INT8, NVFP4, FP8, K8V4, VQ2 and K4V2 KV); less frees KV cache but slows 1-2K-token chunks over long context (`64`: 17-25 % slower attention over 32K-128K keys; `128`: 2-5 %; `192` and up: under 1 %); `0` turns splitting off. See [serving](serving.md) | `256` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash/DFlash2 `1..15` | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--draft-tree-nodes auto\|N` | DFlash2 tree verification: each round verifies a small tree of alternative drafts, built on the device from the drafter's candidate lattice, instead of a single draft, so more drafted tokens are accepted with the same output distribution; any `--kv-dtype`. `auto` (recommended) chooses the single draft or a tree of `draft tokens + 5` or `+ 9` tokens every round from measured speed and acceptance at the current context length, and keeps single drafts (with a warning) on an artifact without tree support; `N` fixes a tree of `draft tokens + 2..32` tokens and keeps seeded runs reproducible, which `auto` does not; see [tree verification](maintainer/tree-verification.md) | off |
| `--draft-tree-paths N` | most branches (root-to-leaf paths) per tree, `2..8` | `8` |
| `--ngram-draft-tokens N` | verified n-gram copy proposals per round beside the `--spec` drafter (Qwen3.8-Flash-Next: also alone, `1..15`), `1..63`; `0` disables; see [ngram copy proposals](ngram.md) | `0` |
| `--ngram-min-match N` | minimum matched tokens for an n-gram proposal, `4..64` | `12` |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--vision-offload auto\|on\|off` | keep the vision tower in pinned system RAM and borrow Device memory only while encoding; takes effect only with `--vision`. `auto` is on for Qwen3.8-Flash-Next (the expert cache lends the encode memory) and off for Qwen3.5 | `auto` |
| `--vision-max-merged N` | merged vision tokens per image or video, `64..32768`; larger media is downscaled during preprocessing | `32768` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--expert-state FILE\|off` | Qwen3.8-Flash-Next: the expert cache's saved state. At startup the engine fills the VRAM expert cache with the experts the file ranks highest (about 1 s for a full cache) and starts their use counts from it; it writes the file when it stops and every 10 minutes between requests (through `FILE.tmp` and a rename). A file from another artifact or of another size is ignored and replaced. `off` starts with an empty cache and saves nothing; cold benchmarks need it | `<artifact>.expert-state` |
| `--chat-template FILE` | use a local Jinja template | artifact template |
| `--no-thinking` | disable thinking | template default |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-loop off\|stop\|conclude` | end the reply or close the thinking when it keeps repeating whole passages | `off` |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | pass an effort value to the selected template | template default |
| `--greedy` | exact argmax decoding | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default; `0` with an output constraint |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |
| `--log-colours on\|off` | colour the statistics output on stderr | on when stderr is a terminal |

When a sampling flag is omitted, Engine selects the general-task preset for the loaded architecture
and rendered prompt mode. The current official models use:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Task-specific profiles such as Qwen's
precise-coding profile use explicit sampling overrides.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/apps/infernix --help` for the exact option contract.

## CUDA synchronization

`INFERNIX_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI
and HTTP server. When unset, it defaults to `spin`, prioritizing low synchronization latency at
the cost of CPU usage while waiting for the GPU. Use `blocking` to let the waiting thread sleep;
the decode performance cost depends on the host. `yield` yields the CPU while waiting, and `auto`
uses CUDA's scheduling heuristic, not an automatic performance benchmark.

For example, `infernix-serve` on an RTX 5090 under Windows, replaying an agentic workload at two
concurrent requests, used 165 % of a CPU core on average with `spin`, 103 % with `yield` and 2 %
with `blocking`. Against `spin`, output tok/s was 1.8 % lower with `yield` and 1.2 % lower with
`blocking`, and mean time to first token moved by at most 0.5 %.

```bash
INFERNIX_CUDA_SYNC=blocking ./build/apps/infernix models/qwen3_8_27b_nvfp4.ninfer --prompt "Hello"
```

The Engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply
the schedule, fail startup. This controls device scheduling (including stream synchronization);
it does not override individual CUDA event creation flags.

## Numerics SM count

`INFERNIX_NUMERICS_SMS=N` pins the multiprocessor count that Ops use for launch decompositions
whose results depend on it: the QSA decode, prompt and selection splits and grids, the wide
experts' persistent grid, and the count every device context reports to its Ops (which includes
the GDN chunked prefill). With the same `N` on GPUs of different SM counts, those Ops produce
bit-identical results on each, as a layer split across two GPUs and its tests require. When unset,
empty or `0`, each device uses its own count, which is the default and the fastest choice. A count
below a device's own leaves SMs idle in those launches.

The value is read once per process for both the CLI and HTTP server. A malformed value, or a count
above the SM count of a device the process opens, fails startup when that device's context is
created.

```bash
INFERNIX_NUMERICS_SMS=70 ./build/apps/infernix model.infernix --prompt "Hello"
```

## Device order

`--device` and `--devices` take CUDA device ordinals in PCI bus order, the order in which
`nvidia-smi` lists the GPUs: before its first CUDA call the engine sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` for its process, for both the CLI and HTTP server. A value already
set in the environment is kept. CUDA's own default numbers the fastest GPU 0, so on a machine with
two different GPUs an ordinal can name another GPU than in builds before this order was selected;
`CUDA_DEVICE_ORDER=FASTEST_FIRST` restores that numbering. With one GPU nothing changes.

## Layer split

`--devices A,B --layer-split K` splits Qwen3.8-Flash-Next's 48 layers across two GPUs: layers
`[0, K)`, with their KV cache, recurrent state and expert cache, on `A`, which also runs the engine;
layers `[K, 48)`, the output head, sampling and the MTP drafter on `B`, which also holds the KV
page allocator. Each GPU gets its own share of the dense weights, its own expert cache sized from
its own free memory (`--vram-headroom-mib` applies to each), and startup checks each GPU's memory
before reading the weights. The routed experts stay in Host RAM once, read by both. The hidden
state crosses from `A` to `B` once per call, through pinned host memory (no peer access is needed).

Prefill, forced tokens (an injected thinking-close control) and decoding run across the split:
plain decode rounds and, with `--spec mtp` or n-gram drafts, the drafter on `B` and verification
rounds. Their logits and greedy output are the unsplit run's bit for bit when both GPUs use the
same launch decompositions (`--devices A,A`, or two different GPUs with
[`INFERNIX_NUMERICS_SMS`](#numerics-sm-count) at the smaller one's SM count). A prompt's calls run
chunk by chunk on `A`, then on `B` (each GPU streams its own layers' experts per chunk, so long
prompts prefill slower than on one GPU, most on a slow `A` link). A round runs `A`'s layers, then
`B`'s, each GPU replaying its own CUDA graphs: `--split-handoff event` (the default) orders `B`
after `A` with a cross-device event; `gate` lets `B` start at once and wait on the GPU, bounded
by `--handoff-timeout-ms` (a round whose state does not arrive in time fails with a recoverable
error and the engine keeps serving).

Each GPU's expert cache follows its own PCIe link. A GPU on a slow link (below 8 GB/s, such as
PCIe 3.0 x4) sends most misses to the CPU (it hands the CPU more of them, at most 32 per layer
call) and fills every frame at startup, not half: from the saved expert state (the last session's
cached experts, then the other experts it used, most used first, then the experts it never used),
or without one every expert of its layers in turn, so a GPU with room for all its layers' experts
holds them all. While requests run it promotes only a trickle: at most one expert per layer for a
few layers per round (6 ms of its link, six experts at 3.1 GB/s), copied once its layers of the
next round are done, while the other GPU computes; so its cache follows a conversation's topics
without its link slowing its own layers. The frames a long prompt borrows for its wide calls get
their experts back afterwards. A faster GPU keeps the dynamic cache of a single GPU. The startup
line of each segment names its link and policy. `INFERNIX_TRICKLE_PROMOTIONS=N` (a measurement
knob) sets the trickle's experts per round; `0` promotes nothing during requests.

The hybrid prefix cache (`infernix-serve` reuses prefixes by default) runs across the split: each
GPU keeps the cached blocks' KV for its own layers on the same pages, and copies its own layers'
KV and recurrent state to and from the pinned Host tier on its own streams, so a resumed turn
waits on each GPU only for its own layers. Snapshots, cached blocks and the `--prefix-cache-file`
keep the whole model's layout, so a file saved by an unsplit run loads into a split one and the
reverse. A resumed request keeps the split's exactness: with the same launch decompositions it
gives the unsplit run's bits.

A split refuses, until each is supported, `--vision`, the
[SSD expert tier](qwen3_8-flash-next.md#ssd-expert-tier),
which needs every expert in Host RAM, and `--layer-split auto` (the default with two devices) or
`@FILE`, which need the split planner: pass `--layer-split K`, with `1 <= K < 48`. `--devices A`
with one ordinal runs exactly as `--device A`. `--devices A,A` runs both segments on one GPU, each
with its own context, one after the other (their work never overlaps on the GPU), with the event
handoff only: a test mode for exactness. `INFERNIX_SPLIT_SYNC=1` (a debugging aid) synchronizes
both GPUs before the host restages a call's or round's inputs. `INFERNIX_SPLIT_TIMING=1` (a
measurement knob, also on one GPU as the split's reference) logs at each request's end, at debug
level, where its decode rounds' time went: the mean round period, the time from launch to
synchronization and each GPU's graph time within it, and the host's drafting and commit; then the
same per verified width (with the draft steps run before the round, from which a verified
column's cost follows), the host's steps between rounds (synchronization, commit, cache update,
drafting, launches) and the drafter's catch-up, idle gap and draft graph on the head GPU.
`INFERNIX_WIDTH_COST=X` (a measurement knob, `0..4`) replaces the one-row draft policy's cost of a
draft in plain rounds (0.38); it changes how many columns rounds verify, never a token.

`INFERNIX_LOGITS_HASH=FILE` (a check for this exactness, for both the CLI and the HTTP server)
appends one line per call or round that computes logits to `FILE`: its kind (`prefill`, `forced`,
`decode` or `verify`), the lane and first position of its first row, its width and logit columns,
and the 64-bit FNV-1a of its FP32 logits. Two runs that must agree
bit for bit, such as a split and an unsplit run of the same prompt, write the same lines. Each such
call waits for its logits, so leave it unset otherwise.

## Context and memory

The official artifacts have a native context limit of 262,144 tokens. The practical allocation
on one RTX 5090 depends on the selected artifact, media workload, output budget, and KV-cache type.
The artifact describes its model configuration and weight representations;
`--rope-yarn-factor F` opts into startup-fixed runtime YaRN with a finite factor in `[1,4]`
(default `1`, native RoPE unchanged). It does not modify the artifact or converter. The factor
extends the allowed context ceiling, but does not enlarge the default `--max-context` or KV pool;
request a larger context explicitly. Memory and selected speculative-backend limits still apply.
This is long-context extrapolation, not a guarantee of answer quality; evaluate your workload.

`--kv-dtype` independently selects runtime KV storage. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
Per token and KV head, K+V take 1024 bytes as `bf16`, 528 as `int8`, 516 as `fp8`, 402 as `k8v4`,
288 as `nvfp4`, 196 as `k4v2` and 132 as `vq2`. `vq2` stores 2-bit vector-quantized K and V (one
16-bit code per 8 rotated values) and `k4v2` 4-bit K with the same 2-bit V; both keep the attention
sinks and the most recent 768 keys exact through a per-sequence INT8 window, about 40 MB per state
slot for the 27B models, which every running request and every Device checkpoint slot holds.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, unified workspace,
and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup Infernix reserves model weights, persistent sequence state, one phase-reused
Program workspace, and a separate CUDA Graph driver allowance. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the registered 16,384-token maximum single item. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.
