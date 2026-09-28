#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace velum::pipeline {

// Pre-extracted prompt-side features. These come from the deferred Python-only
// components (speech tokenizer, campplus, matcha mel), written to disk by
// tests/extract_prompt_features.py (same binary layout the acceptance gate uses
// under wavs/flow_inputs/). The C++ pipeline reads them verbatim — it never
// runs the ONNX frontend or matcha itself.
struct PromptFeatures {
  // Flow prompt speech token (the prompt audio's speech-token sequence), (P,) int32.
  std::vector<int32_t> prompt_tokens;
  // Flow prompt_feat: matcha 80-bin mel of the prompt audio, (mel_len1*80,) float32.
  std::vector<float> prompt_feat;
  // campplus speaker embedding, (192,) float32.
  std::vector<float> spk_embedding;
};

// Top-level CosyVoice3 orchestration: Qwen2Tokenizer -> LLM (generate speech
// tokens) -> Flow (token->mel) -> HiFT (mel->PCM). All three decoder networks run
// on GGML; every weight (LLM, Flow, HiFT) and both fixed buffers are loaded
// exactly once per invocation.
//
// Synthesis is split into two *phases* so the LLM (2.4 GiB of weights) and the
// Flow decoder's DiT graph (~4 GiB, transient, grows with text length) never
// share the GPU — they do not fit an 8 GiB card together.
//
//   Phase 1 (LLM): load_llm() loads the LLM + tokenizer once; generate_tokens()
//     is called once per segment — only the LLM's KV cache / position is reset
//     between segments (no weight reload) — and release_llm() frees the LLM.
//   Phase 2 (vocoder): load_vocoder() loads Flow + HiFT once; decode() turns each
//     segment's speech tokens into PCM audio.
//
// Peak VRAM therefore matches the original release-and-reload design (the LLM
// and the Flow DiT graph are never resident at the same time) while removing the
// per-segment 2.4 GiB weight re-read.
//
// Mode note: this wires the instruction-following (instruct2) flow — the LLM's
// lm_input uses an *empty* prompt-speech-token block (as frontend_instruct2 does),
// while the Flow consumes the prompt's speech token + matcha mel + speaker
// embedding. The voice-cloning branch (LLM prompt_speech_token != empty) is the
// same build_lm_input path with a non-empty vector, already verified in Phase 2.
class Pipeline {
 public:
  Pipeline();
  ~Pipeline();

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // --- Phase 1: LLM (speech-token generation) ---

  // Load the tokenizer data dir and the LLM gguf. Returns false on any error.
  bool load_llm(const std::string& llm_gguf, const std::string& tokenizer_dir);

  // Generate speech tokens for one text in the prompt's voice. `instruct` is the
  // full LLM text prompt (must contain <|endofprompt|>); `seed` drives the LLM
  // sampling RNG. The LLM's weights stay resident across calls; only its
  // autoregressive state (KV cache + position) is reset at the start of each
  // call, so multiple segments are generated back-to-back without reloading
  // weights. Returns false on any error.
  bool generate_tokens(const std::string& instruct, const std::string& text,
                       unsigned seed, std::vector<int32_t>& tokens);

  // Free the LLM weights + backend, ending Phase 1. Must precede load_vocoder().
  void release_llm();

  // --- Phase 2: vocoder (token -> mel -> PCM) ---

  // Load the Flow + HiFT ggufs and the two fixed-asset buffers (HiFT SineGen2
  // source + Flow CFM noise). Returns false on any error.
  bool load_vocoder(const std::string& flow_gguf, const std::string& hift_gguf,
                    const std::string& hift_source_bin,
                    const std::string& flow_noise_bin);

  // Decode one segment's speech tokens to PCM float32 @ 24 kHz (`audio`) and the
  // Flow mel (`mel`, 80*OUT_FRAMES row-major). Returns false on any error.
  bool decode(const std::vector<int32_t>& tokens, const PromptFeatures& prompt,
              std::vector<float>& audio, std::vector<float>& mel);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace velum::pipeline
