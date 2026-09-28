# Velum

A native reimplementation of the [CosyVoice3](https://github.com/QwenAudio/CosyVoice) text-to-speech pipeline with **zero
Python at runtime**. The neural networks (LLM / Flow / HiFT) run on
[GGML](https://github.com/ggml-org/ggml) with CUDA acceleration. Weight
conversion and the acoustic frontend (campplus / speech tokenizer / matcha mel)
are computed **once, offline, in Python** and frozen into files the C++
binary loads verbatim.

**It's designed for better deployment in product as a single executable binary file.**

This project is Human architectured and co-authored by AI.
- LLM: deepseek-v4-pro
- Coding Assistant: Claude Code

## Deps in runtime

```bash
ldd velum
    linux-vdso.so.1 (0x00007ffceb3fd000)
    libicuuc.so.74 => /lib/x86_64-linux-gnu/libicuuc.so.74 (0x00007aab5e400000)
    libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007aab66b91000)
    libcudart.so.12 => /lib/x86_64-linux-gnu/libcudart.so.12 (0x00007aab5e000000)
    libcublas.so.12 => /lib/x86_64-linux-gnu/libcublas.so.12 (0x00007aab57600000)
    libcuda.so.1 => /lib/x86_64-linux-gnu/libcuda.so.1 (0x00007aab51e00000)
    libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007aab51a00000)
    libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007aab5e717000)
    libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007aab66b61000)
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007aab51600000)
    libicudata.so.74 => /lib/x86_64-linux-gnu/libicudata.so.74 (0x00007aab4f800000)
    /lib64/ld-linux-x86-64.so.2 (0x00007aab66c0a000)
    libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007aab66b5a000)
    libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007aab66b55000)
    librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007aab66b50000)
    libcublasLt.so.12 => /lib/x86_64-linux-gnu/libcublasLt.so.12 (0x00007aab2e800000)
```

## Status

All four phases are implemented and verified numerically against the PyTorch
reference:

1. DSP frontend (Whisper 128-bin log-mel + Kaldi 80-bin fbank) — `verify_dsp.py`
2. Flow decoder (PreLookaheadLayer + DiT ×22 + CFM Euler) — `verify_flow.py`
3. HiFT vocoder — `verify_hift.py`
4. LLM backbone (Qwen2-0.5B + CosyVoice3LM heads + ras_sampling) — `verify_llm*.py`

The end-to-end CLI (LLM → Flow → HiFT) is wired and cross-checked by
`tests/verify_e2e.py`. Still **deferred** (pre-extracted by Python): the ONNX
frontend (campplus + speech tokenizer) and the matcha 80-bin mel — the CLI reads
their outputs as files instead of running them.

### Missing features

Please read [Missing Features](docs/missing-features.md), these missings will not be added in the community edition, we offer consulting service for enterprise edition. 
Please contact consulting@hardenedvault.com.

## Quick start

Everything from a fresh clone to a synthesized `.wav`. The only steps that touch
the network or run Python are `pip install` and the `models` target; compiling
`velum` itself is pure CMake/C++.

```sh
# clone (ggml is a git submodule — it must be initialized)
git clone --recurse-submodules https://github.com/hardenedlinux/velum.git
cd velum

# 1. repo venv (torch/numpy) for the one-time weight conversion + RNG buffers
python3 -m venv .venv
.venv/bin/pip install -r tools/requirements-convert.txt

# 2. configure + compile (CUDA auto-detected if a toolkit is present)
cmake -S . -B build
cmake --build build -j

# 3. download & convert models + generate the fixed RNG buffers (~2.6 GB, one-time)
cmake --build build --target models

# 4. synthesize
./build/velum --text "今天天气不错，我们一起去公园散步吧。" --out hello.wav
```

Prerequisites: `git`, CMake ≥ 3.16, a C++17 toolchain, ICU with development
headers (e.g. `libicu-dev` on Debian/Ubuntu), and `python3` + `venv` (only for
steps 1 and 3); a CUDA toolkit is optional. If you already cloned without
submodules, run `git submodule update --init --recursive` before `cmake`.

Each step is explained in detail below (`## Build`, `## Prepare models & assets`,
`## Run`).

## Layout

| dir | purpose |
|---|---|
| `src/dsp/` | Whisper 128-bin log-mel + Kaldi 80-bin fbank (verified) |
| `src/llm/` | Qwen2 autoregressive decoder + CosyVoice3LM heads + ras_sampling (verified) |
| `src/flow/` | DiT flow-matching estimator (verified) |
| `src/hift/` | Causal HiFi-GAN vocoder (verified) |
| `src/pipeline/` | orchestration: tokenizer → LLM → Flow → HiFT |
| `src/cli/` | `velum` end-to-end entry point |
| `src/frontend/` | reserved for the deferred ONNX frontend |
| `tools/` | offline prep: `fetch_model.py`, `convert_weights.py`, `gen_rng_buffers.py`, `export_tokenizer.py`, `gguf.py`, `gen_mel_filters.py` |
| `tests/` | `verify_*.py` numerical checks + `export_*.py` / `extract_prompt_features.py` asset producers |
| `data/` | committed runtime data: `tokenizer/` (BPE vocab + merges), `prompt/` (default prompt-voice bundle) |
| `docs/` | `ARCHITECTURE.md`, ADRs, `DSP.md`, `FLOW.md`, `HIFT.md`, `LLM.md`, `WEIGHT_FORMAT.md`, `annotation-syntax.md`, `missing-features.md` |

## Build

```sh
cmake -S . -B build            # enables CUDA if a toolkit is detected
cmake --build build -j
```

This produces `velum` plus the `velum_*_dump` verification utilities. CUDA is
auto-detected: `ggml`'s CUDA backend is compiled when `VELUM_ENABLE_CUDA=ON`
(default) *and* a CUDA toolchain is found; otherwise it builds CPU-only. Both
backends are linked into `velum` — at runtime it picks CUDA when a device is
present and falls back to CPU. Force the CPU backend with `VELUM_BACKEND=cpu`
(used by the numerical verify scripts so they don't depend on an idle GPU).

> Note: the LLM is large. `llm.gguf` is ~2.6 GB, but synthesis is two-phase: it
> generates every speech token with the LLM, releases it, then loads Flow + HiFT,
> so the LLM and the Flow DiT graph (~4 GiB — the real VRAM peak) never share the
> GPU. If `cudaMalloc` reports out-of-memory, either free the GPU or prefix the
> run with `VELUM_BACKEND=cpu`.

## Prepare models & assets (offline, one-time)

### Model weights & fixed buffers — the `models` target (recommended)

The checkpoints are fetched and converted, and the fixed RNG buffers are
generated, all from one opt-in build target. This is the *only* step that
touches the network or runs Python; the default `cmake --build build` does
neither.

```sh
cmake --build build --target models
```

This (1) downloads `llm.pt` / `flow.pt` / `hift.pt` from the pinned Hugging Face
revision `FunAudioLLM/Fun-CosyVoice3-0.5B-2512@29e01c4e` (falling back to the
ModelScope mirror) and verifies each against a hardcoded SHA-256, (2) converts
them to `models/llm.gguf` / `models/flow.gguf` / `models/hift.gguf`, and (3)
generates the fixed RNG buffers `build/hift_source.bin` /
`build/flow_noise.bin` (see below). Downloads resume on interruption and
already-correct files are skipped, so re-running the target is a no-op.
Conversion and buffer generation need PyTorch — install it once:

```sh
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements-convert.txt
```

(make sure `python3` resolves to that environment when you configure). See
`tools/fetch_model.py` for the weight license status (HF tag `apache-2.0` vs the
model card's "for academic purposes only" note).

The full one-command prepare is:

```sh
cmake --build build && cmake --build build --target models
```

### Fixed RNG buffers (generated, not committed)

Two model-internal buffers are sampled from PyTorch's RNG at construction time
in the reference; the C++ side loads them verbatim (`FlowDecoder::load_noise` /
`HiftVocoder::load_source`) instead of reimplementing PyTorch's RNG:

- `build/flow_noise.bin` — the Flow decoder's CFM seed noise
  `torch.randn([1,80,50*300])` drawn once under `set_all_random_seed(0)`.
- `build/hift_source.bin` — the HiFT `SineGen2` fixed source (`rand_ini` +
  `sine_waves`, `torch.rand`, full 300 s bank).

Neither depends on the weights or on the CosyVoice package, so
`tools/gen_rng_buffers.py` (run by the `models` target) reproduces them with a
bare `torch.manual_seed(0)`: the flow noise is bit-exact, and the HiFT values
are arbitrary noise/phases where *consuming the same bytes* — not matching
CosyVoice's RNG — is what correctness requires (the `verify_e2e.py` reference
overrides its own buffers with this file). They are therefore generated at build
time rather than committed (the HiFT bank is ~247 MiB).

### The manual / offline path

If you already have the checkpoints locally (e.g. a CosyVoice checkout), skip the
download and convert them directly. Two Python environments are used:

- **`.venv`** — repo-local, `torch` + `numpy`, for weight conversion and RNG
  buffer generation.
- **CosyVoice python3.10** — the reference environment that can import
  `cosyvoice`/`transformers`, only for regenerating the committed data
  (tokenizer + prompt-voice bundle) from a model checkout; not needed for
  normal builds.

```sh
# paths used below
MODEL="$HOME/Project/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
PY310="$HOME/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10"
PYTHONPATH="$HOME/Project/CosyVoice/.local/lib/python3.10/site-packages"
```

**1. Convert weights** (`.venv`):

```sh
.venv/bin/python tools/convert_weights.py \
  --llm "$MODEL/llm.pt" --flow "$MODEL/flow.pt" --hift "$MODEL/hift.pt" \
  --out-dir models/
```

writes `models/llm.gguf` / `models/flow.gguf` / `models/hift.gguf` (format-only
conversion, no quantization). Use `--llm "$MODEL/llm.rl.pt"` for the RL-tuned
checkpoint.

**2. Text tokenizer data** (committed — no Python needed):

`vocab.tsv` / `merges.txt` / `added_tokens.tsv` are committed under
`data/tokenizer/` (see `data/tokenizer/README.md` for provenance and licensing).
The build copies them into `build/tokenizer/` automatically, so this step is a
no-op for normal builds. To regenerate them from a model checkout (e.g. when
bumping the CosyVoice version), run the standard-library-only script:

```sh
python3 tools/export_tokenizer.py --out-dir data/tokenizer
```

**3. Generate the fixed RNG buffers** (`.venv`, no CosyVoice needed):

```sh
.venv/bin/python tools/gen_rng_buffers.py --out-dir build/
```

writes `build/hift_source.bin` (HiFT SineGen2 rand_ini + sine_waves) and
`build/flow_noise.bin` (Flow CFM seed noise). These are the model-internal
buffers the reference samples once from PyTorch's RNG at construction; the C++
side loads the frozen values instead of reimplementing the RNG. They do not
depend on the weights and need only torch/numpy, so `tools/gen_rng_buffers.py`
runs without CosyVoice (the `models` target runs it for you). The CosyVoice-env
exporters `tests/export_hift_source.py` / `tests/export_flow_noise.py` remain as
the reference/provenance scripts.

**4. Prompt-voice bundle** (committed — no Python needed):

`prompt_tokens.i32` / `prompt_feat.f32` / `spk_embedding.f32` — the default
zero-shot prompt voice (`asset/zero_shot_prompt.wav`) — are committed under
`data/prompt/` (see `data/prompt/README.md` for provenance and licensing). The
build copies them into `build/prompt/` automatically, so this step is a no-op
for normal builds. To extract a *different* prompt voice (the deferred ONNX
frontend — campplus + speech tokenizer + matcha mel — "temporarily handed to
Python"), run under the CosyVoice python3.10 env:

```sh
"$PY310" tests/extract_prompt_features.py --out-dir data/prompt --prompt-wav <wav>
```

## Run

```sh
./build/velum \
  --text "今天天气不错，我们一起去公园散步吧。" \
  --out hello.wav
```

Model/asset paths default to `models/llm.gguf`, `models/flow.gguf`,
`models/hift.gguf` (the `models` target), `build/hift_source.bin`,
`build/flow_noise.bin`, `build/tokenizer` and `build/prompt` (the committed
default prompt voice, so `--prompt-dir` is optional — pass it to use a different
voice). `--text` is required; `--instruct` is the instruction *body* (default
`请用普通话表达。`); Velum prepends `You are a helpful assistant. ` and appends
`<|endofprompt|>` automatically. Optional dumps:

```sh
./build/velum --text ... --out hello.wav \
  --seed 0 \
  --dump-tokens hello.tokens.i32 \
  --dump-mel    hello.mel.f32 \
  --dump-audio  hello.audio.f32
```

`--seed` drives the LLM sampling RNG; the speech-token sequence is stochastic,
so different seeds (or no `--seed`) give different audio. `VELUM_BACKEND=cpu`
forces CPU.

### Text from a file

`--text-file <path>` reads the input text from a file (`-` = stdin) instead of
the command line, and is equivalent to `--text`:

```sh
./build/velum --text-file input.txt --out hello.wav
```

### Segmented input (per-sentence instructions)

`--segments-file <path>` reads a JSON array of `{"text", "instruct"}` objects
(`-` = stdin). Each segment is synthesized with its own instruction — `instruct`
falls back to `--instruct` when omitted — and the segments are concatenated into
one wav with `--segment-gap-ms` of silence between them (default 200 ms):

```sh
./build/velum --segments-file segments.json --out scene.wav --segments-dir seg
```

```json
[
  {"text": "欢迎来到我们的节目。", "instruct": "请用热情的语气表达。"},
  {"text": "今天我们聊聊天气情况。", "instruct": "请用平静的语气表达。"},
  {"text": "感谢收听，我们下次再见。"}
]
```

`--segments-dir` (optional) also writes each segment's audio to
`segment_001.wav`, `segment_002.wav`, … so they can be used individually. The
annotation syntax usable inside `"text"` (pinyin, CMU phonemes, markers,
`<strong>`) is documented in `docs/annotation-syntax.md`.

## Verify

The `ctest` suite's numerical checks are Python scripts that need the test
dependencies (numpy / torch / torchaudio / whisper / hyperpyyaml) in `.venv`:

```sh
.venv/bin/pip install -r tests/requirements.txt
ctest --test-dir build            # DSP / flow / hift / tokenizer / llm numerical checks
"$PY310" tests/verify_e2e.py      # end-to-end CLI vs PyTorch (CPU, slower)
```

The `ctest` suite needs the reference `.npz` dumps, which are regenerated by the
matching `tests/*_reference.py` scripts (see their docstrings). See
`docs/DSP.md`, `docs/FLOW.md`, `docs/HIFT.md`, `docs/LLM.md` for the measured
error numbers.

## Verification standard

Every cross-check against the PyTorch reference reports a scale-normalized
relative error `rel = max|C++ − ref| / max|ref|` and classifies each stage:

| class | rel err | meaning |
|---|---|---|
| **GREEN** | ≤ 1e-2 (≤ 1%) | numerically correct — matches the reference within float32 accumulation |
| **YELLOW** | 1e-2 … 1e-1 (1%–10%) | above the pass gate; warrants investigation, not yet a proven divergence |
| **RED** | > 1e-1 (> 10%) | structural divergence (wrong op / layout / missing clip) — hard fail |

`GREEN` is the same gate the verify scripts enforce (`fail if rel > 1e-2`);
`YELLOW`/`RED` are escalation bands above it. A `RED` stage is never accepted.

### End-to-end acceptance (`tests/verify_e2e.py`)

The full CLI chain (LLM → Flow → HiFT) cross-checked against the PyTorch
reference, seed 0, `--text "今天天气不错，我们一起去公园散步吧。"`:

| backend | Flow mel (max abs / rel) | HiFT pcm (max abs / rel) | class |
|---|---|---|---|
| CPU | 1.142e-3 / 1.056e-4 | 5.630e-3 / 8.112e-3 | **GREEN** |
| CUDA | 2.220e-3 / 2.052e-4 | 3.495e-3 / 5.036e-3 | **GREEN** |

Both backends are **GREEN**. The CPU HiFT pcm (8.112e-3) sits just inside the
1% line (0.81%) — HiFT's nonlinear (exp/snake/phase) synthesis amplifies the
Flow mel's float32 accumulation (see `docs/HIFT.md`). The CUDA path pins cuBLAS
to `CUBLAS_DEFAULT_MATH` (TF32 disabled, `docs/adr/0002`). Synthesis is two-phase —
the LLM generates every speech token and is released before Flow + HiFT load, so
the resident LLM and the Flow DiT graph (~4 GiB) never share the GPU (they do not
fit an 8 GiB card together). Not a bug; a real divergence would land in `RED`.

The GREEN margin is **sequence-length dependent**: the HiFT max error is
concentrated on a few isolated onset samples and grows with mel length. On a
longer utterance — the ad-copy text with Pronunciation-Inpainting markers
(`<strong>…</strong>`, `[j][ǐ]`), 286 mel frames / 5.72 s — the HiFT pcm max
error is 1.465e-2 (rel 2.427e-2, **YELLOW**), but the mean stays 6.9e-5 and only
17 of 137280 samples (0.012%) exceed 1%. This is the same isolated-onset
accumulation, **not** a marker effect: the Flow/HiFT stages never see the text
(only the tokenizer does, and it is bit-exact on the PI markers).

### CFM flow noise: fixed seed 0 (design choice, not configurable)

The Flow decoder's CFM noise is **fixed at seed 0 by design**. The reference
`CausalConditionalCFM.__init__` samples `rand_noise = torch.randn([1,80,50*300])`
once, under `set_all_random_seed(0)`, and every inference slices
`z = rand_noise[:,:,:n]` from that single frozen buffer. The C++ decoder loads
that exact buffer from `build/flow_noise.bin` (generated by
`tools/gen_rng_buffers.py`, bit-exact with seed-0 `torch.randn`) and reuses it
for **every** synthesis — this is a deliberate, permanent design choice that
keeps the decoder deterministic and reproducible, **not** a configurable option
and **not** a per-run RNG. There is no seed flag for it (see `docs/FLOW.md`).

## Docs

- `docs/ARCHITECTURE.md` — authoritative design.
- `docs/adr/0001-drop-onnx-runtime-for-compute.md` — why ONNX Runtime is frontend-only.
- `docs/DSP.md` / `docs/FLOW.md` / `docs/HIFT.md` / `docs/LLM.md` — per-stage reference + validation numbers.
- `docs/WEIGHT_FORMAT.md` — GGUF tensor organisation.
- `docs/annotation-syntax.md` — text-side control tokens (instruction format, markers, pinyin, CMU phonemes, `<strong>`).
- `docs/missing-features.md` — parity checklist vs. upstream CosyVoice3.
