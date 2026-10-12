# Lightning-Infer

Motore di inferenza C++/CUDA per **Qwen3.8-Flash-Next** (artefatto NVFP4 Dense8 di NVIDIA) su Windows,
con la divisione del modello tra **due GPU diverse**. È un fork di
[Infernix](README.infernix.md), a sua volta derivato da [NInfer](https://github.com/Neroued/ninfer), che
gira su una sola GPU (RTX 5090). Qui è stato portato su Windows e su una macchina con due schede
eterogenee.

> Il codice aggiunto in questo fork è stato scritto con l'AI (Claude Opus 5.5). Infernix originale è
> descritto in [README.infernix.md](README.infernix.md); la licenza resta Apache 2.0 ([LICENSE](LICENSE)).

## Cosa c'è di nuovo rispetto a Infernix

- **Windows nativo**: build con MSVC 2022 e CUDA 13.3. Correzioni per il compilatore (lambda e
  costanti dei kernel, un crash di `cicc`), RAM fissata a blocchi da 8 GiB (Windows rifiuta un blocco
  unico oltre ~48 GiB) e misura del collegamento PCIe robusta (dopo un grosso blocco di RAM fissata
  Windows deprime i trasferimenti, e una scheda inattiva parte a Gen1).
- **Due GPU eterogenee** (`--devices 0,1 --layer-split K`): i primi K strati sulla prima scheda, il
  resto più la testa e l'MTP sulla seconda. Funziona senza P2P: tra le schede passa solo il residuo
  (20 KiB per token) attraverso la RAM. Ogni scheda ha la sua cache degli esperti, la sua parte di
  KV e il suo servizio CPU per gli esperti mancanti.
- **Uscita identica bit per bit**: le risposte divise sono uguali byte per byte a quelle su una scheda
  sola (con `INFERNIX_NUMERICS_SMS=70` le riduzioni usano lo stesso ordine su schede con un numero di
  SM diverso). Perplessità invariata: 3,6792 sul test di riferimento.
- **Cache dei prefissi tra le due schede**: istantanee e KV per segmento, file compatibili con la
  versione non divisa.
- **Lettura dei prompt lunghi sulle due schede**: ogni strato legge tutto il prompt in un colpo
  (gli esperti si caricano una volta per strato), la seconda scheda calcola gli esperti mancanti della
  prima durante la lettura e le copie sono in pipeline.
- **Kernel ritarati** per i chip GB203 (84 e 70 SM invece dei 170 della 5090), con risultati identici.
- **Cache degli esperti adattiva** sulla scheda col collegamento lento (promozioni "a goccia" a ogni
  round) e bozze n-gram che saltano l'MTP quando la copia è già verificata.
- **Visione sulle due schede**: l'encoder gira sulla scheda col collegamento più veloce, gli embedding
  vengono copiati dove servono. Immagini e video identici alla versione non divisa.

## Risultati

Macchina: RTX PRO 4000 Blackwell (24 GB, PCIe Gen3 x4 dal chipset) + RTX 5080 (16 GB, Gen4 x16),
Ryzen 9 5900X, 128 GB DDR4-3200, Windows 11. Divisione a 28 strati sulla PRO 4000.
Conversazione di 20 turni su temi diversi con codice incollato (~50K token), media della seconda metà.

| | Generazione (tok/s) | Picchi | Lettura di 50K token |
|---|---|---|---|
| **Lightning-Infer, due GPU** | **80-90** | 120-145 | **~21 s** |
| Lightning-Infer, sessione da agente di codice (bozze n-gram) | 103-107 | | |
| Infernix, sola RTX 5080 | 46-49 | | |
| Strata (Orca Q4_K_S), due GPU | 34 | 63 | 43-78 s per incolla |

Contesto lungo (512K con YaRN x2, finestra riempita a passi di 50K): 69 tok/s a 63K, 62 a 257K,
55 a 514K.

## Build (Windows)

Requisiti: Visual Studio 2022 Build Tools (MSVC 14.44), CUDA 13.3, CMake + Ninja, vcpkg in
`C:\vcpkg` (ffmpeg e curl, triplet `x64-windows`).

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
cmake -S . -B build-windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a ^
      -DINFERNIX_BUILD_APPS=ON
cmake --build build-windows --parallel 8
```

I binari finiscono in `build-windows\bin` (`infernix-serve.exe`, `infernix.exe` e le DLL di ffmpeg/curl).

## Avvio

```bat
set CUDA_DEVICE_ORDER=PCI_BUS_ID
infernix-serve.exe <artefatto>.infernix --ngram-volume <volume>.ngram ^
  --devices 0,1 --layer-split 28 --max-context 262144 ^
  --kv-dtype int8 --prefill-chunk 4096 --spec mtp --draft-tokens 4 --lm-head-draft ^
  --ngram-draft-tokens 7 --chat-template tools\chat_templates\qwen3_8.jinja ^
  --vision --tolerant-tool-calls --model-id qwen3.8-flash-next --host 127.0.0.1 --port 8891
```

- `--devices 0,1 --layer-split 28`: strati 0-27 sulla scheda 0, 28-47 + testa + MTP sulla scheda 1.
  Con `CUDA_DEVICE_ORDER=PCI_BUS_ID` gli ordinali seguono il bus PCI.
- `--chat-template`: il template mantenuto da Infernix accetta i messaggi `developer` e le
  `instructions` che manda Codex (quello incorporato nell'artefatto li rifiuta).
- Per 512K di contesto: `--max-context 524288 --rope-yarn-factor 2`.
- Servono ~71 GB di RAM fissata per gli esperti.

API: OpenAI Chat Completions e Responses, Anthropic Messages (vedi [docs/serving.md](docs/serving.md)).
Provato con opencode, pi, Codex e richieste in stile Claude Code.

## Rami

| Ramo | Contenuto |
|---|---|
| `wave1` | ramo principale: Windows + due GPU + visione |
| `master` | Infernix originale con le sole correzioni per Windows (una GPU) |
| `mgpu`, `w5a`, `w5b`, `w5c` | rami di lavoro già uniti in `wave1` |

## Documentazione

- [README.infernix.md](README.infernix.md): Infernix originale (funzioni, modelli, benchmark sulla 5090)
- [docs/](docs/README.md): CLI, server, guida a Qwen3.8-Flash-Next
- [docs/maintainer/qwen3_8-flash-next-design.md](docs/maintainer/qwen3_8-flash-next-design.md):
  progetto del motore, compresa la visione sulle due GPU (§19.3.2)
