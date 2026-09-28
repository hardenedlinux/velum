# Prompt-voice bundle (committed)

These three files are the *runtime* prompt-side tensors the C++ `Pipeline`
reads for zero-shot synthesis — the prompt voice's speech tokens, its matcha
mel features, and its speaker embedding. They are committed so that building and
running Velum needs no Python and no access to the ONNX frontend or the original
model: `cmake --build build` copies them into `build/prompt/` (the default
`--prompt-dir`).

| file | content |
|---|---|
| `prompt_tokens.i32` | prompt speech tokens, int32 (87 tokens) |
| `prompt_feat.f32` | prompt matcha mel, float32 (174 × 80 = 2·token_len × 80) |
| `spk_embedding.f32` | campplus speaker embedding, float32 (192 dims) |

These are the three tensors `CosyVoiceFrontEnd.frontend_zero_shot` produces from
the prompt wav; the C++ CLI reads them instead of running campplus / the speech
tokenizer / matcha mel (that ONNX frontend is deferred — see
`docs/ARCHITECTURE.md`).

## Provenance & source verification

Pre-extracted by `tests/extract_prompt_features.py` from the CosyVoice3 default
zero-shot prompt voice (`asset/zero_shot_prompt.wav` in the CosyVoice repo) using
the frontend model files `campplus.onnx`, `speech_tokenizer_v3.onnx` and
`spk2info.pt` (plus `CosyVoice-BlankEN` for the tokenizer). The extraction is
deterministic: the prompt-side tensors depend only on the prompt wav, never on
the LLM/Flow/HiFT decoders or any RNG.

| | value |
|---|---|
| HF model | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| HF revision | `29e01c4e8d000f4bcd70751be16fa94bf3d85a18` |
| CosyVoice commit | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` |
| Prompt voice | `asset/zero_shot_prompt.wav` (CosyVoice3 demo prompt, 24 kHz) |
| Generated | 2026-09-28 |

## Output artifacts (committed)

| file | SHA-256 |
|---|---|
| `prompt_tokens.i32` | `2a404d932260f43f89cb1cef39093ec723b765b04a48030e34e8236dafcdf7a4` |
| `prompt_feat.f32` | `2b942fba28fd46edc7e454f03e7ebca3d2a4be4e29b0be0f9fdadac16c114cac` |
| `spk_embedding.f32` | `661dec10603100646bc68c986b964da16801d17e2eb281defe4a789ea256abe9` |

Regenerate with `tests/extract_prompt_features.py --out-dir data/prompt` under
the CosyVoice python3.10 env (see that script's docstring for the exact paths);
pass `--prompt-wav <wav>` to pre-extract a different voice.

## License

These files are feature vectors derived from CosyVoice3's default zero-shot
prompt voice and from the model files of the Hugging Face repository
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (tagged `apache-2.0`). The model card's
`Disclaimer` section additionally states that "The content provided above is for
academic purposes only and is intended to demonstrate technical capabilities".
The maintainers have not yet clarified this tension (Hugging Face discussion #19,
"Clarification on Commercial Use", open at the time of writing). The above
records the facts only and is not a legal conclusion.
