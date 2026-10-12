# Lightning-Infer

**Lightning-Infer è un motore di inferenza locale per Qwen3.8-Flash-Next che usa insieme due GPU
NVIDIA diverse sullo stesso PC**, su Windows, per farlo girare veloce come un modello molto più
piccolo: **80-90 token/s** in conversazione, **oltre 100** quando lavora sul codice, con contesto fino
a 512K token, immagini e video.

Qwen3.8-Flash-Next è un modello mixture-of-experts da 24.576 esperti (circa 70 GB in NVFP4): non
entra in nessuna scheda consumer. Lightning-Infer lo divide per strati tra le due GPU, tiene gli
esperti in RAM, mette in VRAM quelli che il modello sta usando davvero e li aggiorna mentre la
conversazione cambia argomento.

## Perché

- **Due GPU qualsiasi, anche diverse.** Una scheda grande e una piccola, collegate in modo diverso
  (anche una su uno slot lento del chipset), senza NVLink né P2P. Ogni scheda tiene i suoi strati, la
  sua parte di contesto e la sua cache di esperti; tra le due passano solo 20 KiB per token.
- **Nessuna perdita di qualità.** Il risultato è identico bit per bit a quello calcolato su una scheda
  sola: stesse risposte, stessa perplessità. Ogni ottimizzazione è verificata così prima di entrare.
- **Pensato per gli agenti di codice.** Bozze MTP e n-gram (sul codice che un agente rilegge e
  riscrive accettano lunghe sequenze), cache dei prompt che riusa la conversazione già letta, chiamate
  agli strumenti robuste.
- **Prompt lunghi letti in fretta.** 50K token di codice incollato vengono letti in circa 21 secondi.
- **Immagini e video** in ogni richiesta, senza costi sulla velocità del testo.
- **Compatibile con i client che già usi.** API OpenAI (Chat Completions e Responses) e Anthropic
  Messages: provato con Codex, opencode, pi e richieste in stile Claude Code.

## Prestazioni

Misurate su RTX PRO 4000 Blackwell (24 GB, PCIe Gen3 x4) + RTX 5080 (16 GB, Gen4 x16),
Ryzen 9 5900X, 128 GB DDR4-3200, Windows 11. Conversazione di 20 turni con codice incollato
(~50K token), media della seconda metà.

| | Generazione (tok/s) | Picchi | Lettura di 50K token |
|---|---|---|---|
| **Lightning-Infer, conversazione** | **80-90** | 120-145 | **~21 s** |
| **Lightning-Infer, agente di codice** | **103-107** | | |
| Strata (Orca Q4_K_S) sulle stesse due GPU | 34 | 63 | 43-78 s |

Con la finestra di 512K riempita a passi di 50K token: 69 tok/s a 63K, 62 a 257K, 55 a 514K.

## Requisiti

- Due GPU NVIDIA Blackwell (RTX 50 / RTX PRO serie Blackwell), oppure una sola.
- ~71 GB di RAM libera per gli esperti (128 GB consigliati).
- Un SSD NVMe per il volume n-gram da 52 GB.
- Windows 11, driver NVIDIA recente.
- Il modello: artefatto Qwen3.8-Flash-Next NVFP4 Dense8 (`.infernix`) e il suo volume `.ngram`.

## Build

Servono Visual Studio 2022 Build Tools (MSVC 14.44), CUDA 13.3, CMake + Ninja e vcpkg in `C:\vcpkg`
(ffmpeg e curl, triplet `x64-windows`).

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
cmake -S . -B build-windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a ^
      -DINFERNIX_BUILD_APPS=ON
cmake --build build-windows --parallel 8
```

I programmi finiscono in `build-windows\bin`: `infernix-serve.exe` (server) e `infernix.exe` (riga di
comando).

## Avvio

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
infernix-serve.exe <modello>.infernix --ngram-volume <volume>.ngram ^
  --devices 0,1 --layer-split 28 --max-context 262144 ^
  --kv-dtype int8 --prefill-chunk 4096 --spec mtp --draft-tokens 4 --lm-head-draft ^
  --ngram-draft-tokens 7 --chat-template tools\chat_templates\qwen3_8.jinja ^
  --vision --tolerant-tool-calls --model-id qwen3.8-flash-next --host 127.0.0.1 --port 8891
```

- `--devices 0,1 --layer-split 28`: strati 0-27 sulla GPU 0, strati 28-47 e la testa sulla GPU 1.
  Conviene mettere più strati sulla scheda con più VRAM; con `CUDA_DEVICE_ORDER=PCI_BUS_ID` i numeri
  delle GPU seguono l'ordine degli slot.
- `--device N` al posto di `--devices` usa una GPU sola.
- 512K di contesto: `--max-context 524288 --rope-yarn-factor 2`.
- `--chat-template` serve a Codex (messaggi `developer` e `instructions`).

Poi punta il client a `http://127.0.0.1:8891/v1` con il modello `qwen3.8-flash-next`.

## Documentazione

- [docs/serving.md](docs/serving.md): server e API
- [docs/cli.md](docs/cli.md): riga di comando
- [docs/qwen3_8-flash-next.md](docs/qwen3_8-flash-next.md): guida al modello e regolazioni

## Crediti e licenza

Lightning-Infer nasce da [Infernix](README.infernix.md), a sua volta derivato da
[NInfer](https://github.com/Neroued/ninfer) di Neroued. Licenza Apache 2.0 ([LICENSE](LICENSE)).
Il codice di Lightning-Infer è stato scritto con l'aiuto dell'AI (Claude Opus 5.5).
