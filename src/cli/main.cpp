// velum — native (zero-Python-at-runtime) CosyVoice3 TTS.
//
// End-to-end synthesis: text (+ prompt voice) -> PCM WAV, running the LLM,
// Flow and HiFT decoders on GGML. The prompt's speech token, matcha mel and
// speaker embedding are *pre-extracted* (Python) into a bundle directory — the
// ONNX frontend (campplus + speech tokenizer) and matcha mel are deferred, so
// this CLI reads them as files rather than computing them.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "velum/flow/flow.h"
#include "velum/frontend/frontend.h"
#include "velum/hift/hift.h"
#include "velum/llm/llm.h"
#include "velum/pipeline/pipeline.h"

namespace {

struct Args {
  std::string instruct = "请用普通话表达。";  // instruction body; Velum adds the prefix + <|endofprompt|>
  std::string text;
  std::string text_file;       // --text-file: read `text` from a file ("-" = stdin)
  std::string segments_file;   // --segments-file: JSON array of {text, instruct}
  std::string prompt_dir = "build/prompt";
  std::string llm_gguf = "models/llm.gguf";
  std::string flow_gguf = "models/flow.gguf";
  std::string hift_gguf = "models/hift.gguf";
  std::string hift_source = "build/hift_source.bin";
  std::string flow_noise = "build/flow_noise.bin";
  std::string tokenizer_dir = "build/tokenizer";
  std::string out_wav;
  std::string dump_tokens;
  std::string dump_mel;
  std::string dump_audio;
  int segment_gap_ms = 200;    // --segment-gap-ms: inter-segment silence
  std::string segments_dir;    // --segments-dir: optional per-segment wav output
  unsigned seed = 0;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
    "usage: %s (--text <str> | --text-file <path> | --segments-file <path>)\n"
    "           --prompt-dir <dir> --out <wav> [options]\n"
    "\n"
    "  input (exactly one):\n"
    "  --text <str>           text to synthesize\n"
    "  --text-file <path>     read the text from a file ('-' = stdin)\n"
    "  --segments-file <path> JSON array [{\"text\":\"...\",\"instruct\":\"...\"}, ...]\n"
    "                         ('-' = stdin). `instruct` is the instruction *body*\n"
    "                         (optional, falls back to --instruct); synthesized\n"
    "                         into one wav with silence between segments.\n"
    "\n"
    "  --prompt-dir <dir>   pre-extracted prompt bundle (default build/prompt):\n"
    "                         prompt_tokens.i32  (P int32 speech tokens)\n"
    "                         prompt_feat.f32    (mel_len1*80 matcha mel)\n"
    "                         spk_embedding.f32  (192 campplus embedding)\n"
    "  --out <wav>          output 16-bit PCM WAV @ 24 kHz (required)\n"
    "\n"
    "  --instruct <str>     instruction body (Velum adds the \"You are a helpful\n"
    "                         assistant. \" prefix and <|endofprompt|>); also the\n"
    "                         fallback for --segments-file entries lacking one\n"
    "  --segment-gap-ms <n> inter-segment silence for --segments-file (default 200)\n"
    "  --segments-dir <dir> also write each segment's wav (segment_001.wav, ...)\n"
    "  --seed <n>           LLM sampling seed (default 0)\n"
    "  --llm/--flow/--hift <gguf>         model files (default models/*.gguf)\n"
    "  --hift-source <bin>  SineGen2 source asset (default build/hift_source.bin)\n"
    "  --flow-noise <bin>   CFM noise asset (default build/flow_noise.bin)\n"
    "  --tokenizer-dir <dir>              (default build/tokenizer)\n"
    "  --dump-tokens <i32>  also write generated speech tokens (int32)\n"
    "  --dump-mel <f32>     also write the Flow mel (80*OUT_FRAMES float32)\n"
    "  --dump-audio <f32>   also write the raw PCM (L_s float32, before 16-bit\n"
    "                       quantisation) for numerical comparison\n",
    argv0);
}

bool parse_args(int argc, char** argv, Args* a) {
  auto need = [&](int& i) -> const char* {
    if (i + 1 >= argc) return nullptr;
    return argv[++i];
  };
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    const char* v;
    if (k == "--text")           { v = need(i); if (!v) return false; a->text = v; }
    else if (k == "--text-file") { v = need(i); if (!v) return false; a->text_file = v; }
    else if (k == "--segments-file"){ v = need(i); if (!v) return false; a->segments_file = v; }
    else if (k == "--segment-gap-ms"){ v = need(i); if (!v) return false; a->segment_gap_ms = (int)std::strtol(v, nullptr, 10); }
    else if (k == "--segments-dir"){ v = need(i); if (!v) return false; a->segments_dir = v; }
    else if (k == "--prompt-dir"){ v = need(i); if (!v) return false; a->prompt_dir = v; }
    else if (k == "--out")       { v = need(i); if (!v) return false; a->out_wav = v; }
    else if (k == "--instruct")  { v = need(i); if (!v) return false; a->instruct = v; }
    else if (k == "--seed")      { v = need(i); if (!v) return false; a->seed = (unsigned)std::strtoul(v, nullptr, 10); }
    else if (k == "--llm")       { v = need(i); if (!v) return false; a->llm_gguf = v; }
    else if (k == "--flow")      { v = need(i); if (!v) return false; a->flow_gguf = v; }
    else if (k == "--hift")      { v = need(i); if (!v) return false; a->hift_gguf = v; }
    else if (k == "--hift-source"){ v = need(i); if (!v) return false; a->hift_source = v; }
    else if (k == "--flow-noise"){ v = need(i); if (!v) return false; a->flow_noise = v; }
    else if (k == "--tokenizer-dir"){ v = need(i); if (!v) return false; a->tokenizer_dir = v; }
    else if (k == "--dump-tokens"){ v = need(i); if (!v) return false; a->dump_tokens = v; }
    else if (k == "--dump-mel")  { v = need(i); if (!v) return false; a->dump_mel = v; }
    else if (k == "--dump-audio"){ v = need(i); if (!v) return false; a->dump_audio = v; }
    else return false;
  }
  if (a->segment_gap_ms < 0) a->segment_gap_ms = 0;
  return !a->out_wav.empty();
}

bool read_i32(const std::string& path, std::vector<int32_t>& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0 || n % 4 != 0) { std::fclose(f); return false; }
  out.resize(n / 4);
  bool ok = std::fread(out.data(), 4, out.size(), f) == out.size();
  std::fclose(f);
  return ok;
}

bool read_f32(const std::string& path, std::vector<float>& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0 || n % 4 != 0) { std::fclose(f); return false; }
  out.resize(n / 4);
  bool ok = std::fread(out.data(), 4, out.size(), f) == out.size();
  std::fclose(f);
  return ok;
}

bool write_i32(const std::string& path, const std::vector<int32_t>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = std::fwrite(v.data(), 4, v.size(), f) == v.size();
  std::fclose(f);
  return ok;
}

bool write_f32(const std::string& path, const std::vector<float>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = std::fwrite(v.data(), 4, v.size(), f) == v.size();
  std::fclose(f);
  return ok;
}

bool write_wav(const std::string& path, const std::vector<float>& audio, int sr) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;

  auto put32 = [&](uint32_t x) { std::fwrite(&x, 4, 1, f); };
  auto put16 = [&](uint16_t x) { std::fwrite(&x, 2, 1, f); };

  const uint32_t n = (uint32_t)audio.size();
  const uint32_t data_bytes = n * 2;
  // 44-byte canonical WAV header (mono, 16-bit PCM).
  std::fwrite("RIFF", 1, 4, f); put32(36 + data_bytes);
  std::fwrite("WAVE", 1, 4, f);
  std::fwrite("fmt ", 1, 4, f); put32(16); put16(1); put16(1); put32(sr);
  put32(sr * 2); put16(2); put16(16);
  std::fwrite("data", 1, 4, f); put32(data_bytes);

  for (float s : audio) {
    float v = s * 32767.0f;
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    put16((uint16_t)(int16_t)v);
  }
  std::fclose(f);
  return true;
}

// Read an entire stream (stdin or a regular file) into `out`; false on I/O error.
bool read_stream(FILE* f, std::string* out) {
  out->clear();
  char buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
  return !std::ferror(f);
}

// Read a whole text input from `path`, or from stdin when path == "-".
bool read_input(const std::string& path, std::string* out) {
  if (path == "-") return read_stream(stdin, out);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  const bool ok = read_stream(f, out);
  std::fclose(f);
  return ok;
}

// Wrap a user-supplied instruction *body* into the full CosyVoice3 instruct2
// prompt. The caller writes only the instruction itself (e.g. "请用普通话表达。");
// Velum prepends the fixed "You are a helpful assistant. " prefix and appends
// "<|endofprompt|>" so the result is byte-identical to the upstream presets
// (cosyvoice/utils/common.py instruct_list). If the caller already embedded the
// prefix or <|endofprompt|>, that is an error (reported, never silently stripped).
bool wrap_instruct(const std::string& body, const std::string& ctx,
                   std::string* out) {
  const std::string label = ctx.empty() ? "--instruct" : ctx;
  if (body.find("You are a helpful assistant") != std::string::npos) {
    std::fprintf(stderr,
      "velum: %s already contains the \"You are a helpful assistant\" prefix; "
      "write only the instruction body (e.g. \"请用普通话表达。\")\n", label.c_str());
    return false;
  }
  if (body.find("<|endofprompt|>") != std::string::npos) {
    std::fprintf(stderr,
      "velum: %s already contains <|endofprompt|>; Velum appends it automatically\n",
      label.c_str());
    return false;
  }
  *out = "You are a helpful assistant. " + body + "<|endofprompt|>";
  return true;
}

// One --segments-file entry with its instruct already wrapped into the full
// prompt (the JSON "instruct" body, or the global --instruct fallback).
struct Segment {
  std::string text;
  std::string instruct;
};

// One unit of synthesis work. During Phase 1 (LLM) each job's `tokens` are
// generated; during Phase 2 (vocoder) its `audio` + `mel` are produced. Jobs are
// filled in two passes so the LLM (2.4 GiB) and the Flow DiT graph (~4 GiB) never
// coexist in VRAM.
struct Job {
  std::string instruct;
  std::string text;
  std::vector<int32_t> tokens;
  std::vector<float> audio;
  std::vector<float> mel;
};

// Parse a --segments-file JSON array of {"text", "instruct"} objects. `instruct`
// is optional per entry and holds an instruction *body* that is wrapped here via
// wrap_instruct(); entries without one fall back to `default_instruct` (already
// wrapped). `text` is passed through verbatim (Velum does not parse CosyVoice3
// annotations). On any error prints a message naming the offending segment
// (1-based) and returns false.
bool parse_segments(const std::string& raw, const std::string& default_instruct,
                    std::vector<Segment>* out) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(raw);
  } catch (const nlohmann::json::parse_error& e) {
    std::fprintf(stderr, "velum: --segments-file: invalid JSON: %s\n", e.what());
    return false;
  }
  if (!j.is_array()) {
    std::fprintf(stderr,
      "velum: --segments-file: expected a JSON array of {\"text\", \"instruct\"} objects\n");
    return false;
  }
  out->clear();
  out->reserve(j.size());
  for (size_t i = 0; i < j.size(); i++) {
    const size_t idx = i + 1;  // 1-based, for user-facing messages
    const nlohmann::json& e = j[i];
    if (!e.is_object()) {
      std::fprintf(stderr, "velum: --segments-file: segment %zu is not an object\n", idx);
      return false;
    }
    for (auto it = e.begin(); it != e.end(); ++it) {
      if (it.key() != "text" && it.key() != "instruct") {
        std::fprintf(stderr, "velum: --segments-file: segment %zu has unknown field \"%s\"\n",
                     idx, it.key().c_str());
        return false;
      }
    }
    if (!e.contains("text")) {
      std::fprintf(stderr, "velum: --segments-file: segment %zu is missing \"text\"\n", idx);
      return false;
    }
    if (!e["text"].is_string()) {
      std::fprintf(stderr, "velum: --segments-file: segment %zu \"text\" must be a string\n", idx);
      return false;
    }
    std::string instruct = default_instruct;  // already wrapped
    if (e.contains("instruct")) {
      if (!e["instruct"].is_string()) {
        std::fprintf(stderr, "velum: --segments-file: segment %zu \"instruct\" must be a string\n", idx);
        return false;
      }
      char ctx[64];
      std::snprintf(ctx, sizeof(ctx), "segment %zu \"instruct\"", idx);
      if (!wrap_instruct(e["instruct"].get<std::string>(), ctx, &instruct)) return false;
    }
    out->push_back({e["text"].get<std::string>(), instruct});
  }
  return true;
}

// Short linear fade-in/out at the head and tail of `audio` (used to de-click
// segment boundaries when concatenating). `fade` samples at each edge; clamped
// so the two fades never overlap on a very short clip.
void fade_edges(std::vector<float>& audio, size_t fade) {
  if (fade == 0 || audio.empty()) return;
  const size_t n = audio.size();
  const size_t f = std::min(fade, n / 2);
  if (f < 2) return;
  for (size_t i = 0; i < f; i++) {
    const float g = (float)i / (float)(f - 1);  // linear 0..1
    audio[i] *= g;
    audio[n - 1 - i] *= g;
  }
}

// Concatenate per-segment audio into one wav, separated by `--segment-gap-ms`
// of silence with a short de-click fade at each segment edge. Optionally writes
// each segment's raw (pre-fade) audio to `--segments-dir`.
int write_segments_output(const Args& a, std::vector<Job>& jobs) {
  const int sr = 24000;
  const size_t fade = (size_t)(5 * sr / 1000);                    // 5 ms de-click
  const size_t gap = (size_t)((int64_t)a.segment_gap_ms * sr / 1000);

  if (!a.segments_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(a.segments_dir, ec);
    if (ec) {
      std::fprintf(stderr, "velum: failed to create --segments-dir %s\n",
                   a.segments_dir.c_str());
      return 1;
    }
  }

  std::vector<float> combined;
  for (size_t s = 0; s < jobs.size(); s++) {
    if (s > 0) combined.insert(combined.end(), gap, 0.0f);  // inter-segment silence

    // Optional per-segment wav: the raw synthesis, before the de-click fade.
    if (!a.segments_dir.empty()) {
      char name[64];
      std::snprintf(name, sizeof(name), "segment_%03zu.wav", s + 1);
      const std::string path = a.segments_dir + "/" + name;
      if (!write_wav(path, jobs[s].audio, sr)) {
        std::fprintf(stderr, "velum: failed to write %s\n", path.c_str());
        return 1;
      }
    }

    fade_edges(jobs[s].audio, fade);
    combined.insert(combined.end(), jobs[s].audio.begin(), jobs[s].audio.end());
  }

  if (!write_wav(a.out_wav, combined, sr)) {
    std::fprintf(stderr, "velum: failed to write %s\n", a.out_wav.c_str());
    return 1;
  }
  std::fprintf(stderr,
    "velum: wrote %s  segments=%zu samples=%zu (%.2f s @ %d Hz)\n",
    a.out_wav.c_str(), jobs.size(), combined.size(),
    (double)combined.size() / sr, sr);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, &a)) {
    usage(argv[0]);
    return 2;
  }

  // Exactly one of --text / --text-file / --segments-file must be given.
  const int input_kinds =
      (a.text.empty() ? 0 : 1) + (a.text_file.empty() ? 0 : 1) +
      (a.segments_file.empty() ? 0 : 1);
  if (input_kinds != 1) {
    std::fprintf(stderr,
      "velum: specify exactly one of --text, --text-file, --segments-file\n");
    usage(argv[0]);
    return 2;
  }

  // Wrap the global instruction body into the full instruct2 prompt once, up
  // front (fail fast before loading models). Segments with their own "instruct"
  // are wrapped inside parse_segments(); those without fall back to this.
  std::string instruct_full;
  if (!wrap_instruct(a.instruct, "", &instruct_full)) return 2;

  // Resolve the full input up front (before loading models) so a malformed
  // --segments-file fails fast. Either a multi-segment list or a single text.
  std::vector<Segment> segments;
  std::string text = a.text;
  if (!a.segments_file.empty()) {
    std::string raw;
    if (!read_input(a.segments_file, &raw)) {
      std::fprintf(stderr, "velum: failed to read --segments-file %s\n",
                   a.segments_file.c_str());
      return 1;
    }
    if (!parse_segments(raw, instruct_full, &segments)) return 1;
    if (segments.empty()) {
      std::fprintf(stderr, "velum: --segments-file contains no segments\n");
      return 1;
    }
  } else if (!a.text_file.empty()) {
    if (!read_input(a.text_file, &text)) {
      std::fprintf(stderr, "velum: failed to read --text-file %s\n",
                   a.text_file.c_str());
      return 1;
    }
    if (text.empty()) {
      std::fprintf(stderr, "velum: --text-file %s is empty\n", a.text_file.c_str());
      return 1;
    }
  }

  velum::pipeline::PromptFeatures prompt;
  if (!read_i32(a.prompt_dir + "/prompt_tokens.i32", prompt.prompt_tokens) ||
      !read_f32(a.prompt_dir + "/prompt_feat.f32", prompt.prompt_feat) ||
      !read_f32(a.prompt_dir + "/spk_embedding.f32", prompt.spk_embedding)) {
    std::fprintf(stderr, "velum: failed to read prompt bundle in %s\n", a.prompt_dir.c_str());
    return 1;
  }
  if (prompt.spk_embedding.size() != 192) {
    std::fprintf(stderr, "velum: spk_embedding.f32 has %zu floats, expected 192\n",
                 prompt.spk_embedding.size());
    return 1;
  }

  // Build the ordered job list: one job for single-text input, or one per
  // --segments-file entry.
  std::vector<Job> jobs;
  if (!a.segments_file.empty()) {
    jobs.reserve(segments.size());
    for (const auto& s : segments) jobs.push_back({s.instruct, s.text, {}, {}, {}});
  } else {
    jobs.push_back({instruct_full, text, {}, {}, {}});
  }

  velum::pipeline::Pipeline pipe;

  // Phase 1 (LLM): load the LLM once, generate every segment's speech tokens
  // back-to-back (only llm.reset() between segments), then release its weights
  // so they never coexist with the Flow DiT graph in VRAM.
  if (!pipe.load_llm(a.llm_gguf, a.tokenizer_dir)) {
    std::fprintf(stderr, "velum: LLM load failed\n");
    return 1;
  }
  for (size_t i = 0; i < jobs.size(); i++) {
    if (!pipe.generate_tokens(jobs[i].instruct, jobs[i].text, a.seed, jobs[i].tokens)) {
      std::fprintf(stderr, "velum: LLM generation failed for segment %zu\n", i + 1);
      return 1;
    }
  }
  pipe.release_llm();

  // Phase 2 (vocoder): load Flow + HiFT once, decode every segment's tokens.
  if (!pipe.load_vocoder(a.flow_gguf, a.hift_gguf, a.hift_source, a.flow_noise)) {
    std::fprintf(stderr, "velum: vocoder load failed\n");
    return 1;
  }
  for (size_t i = 0; i < jobs.size(); i++) {
    if (!pipe.decode(jobs[i].tokens, prompt, jobs[i].audio, jobs[i].mel)) {
      std::fprintf(stderr, "velum: decode failed for segment %zu\n", i + 1);
      return 1;
    }
  }

  // Multi-segment input: concatenate into one wav.
  if (jobs.size() > 1) return write_segments_output(a, jobs);

  // Single-segment input.
  const Job& r = jobs[0];
  if (!write_wav(a.out_wav, r.audio, 24000)) {
    std::fprintf(stderr, "velum: failed to write %s\n", a.out_wav.c_str());
    return 1;
  }
  if (!a.dump_tokens.empty()) write_i32(a.dump_tokens, r.tokens);
  if (!a.dump_mel.empty()) write_f32(a.dump_mel, r.mel);
  if (!a.dump_audio.empty()) write_f32(a.dump_audio, r.audio);

  std::fprintf(stderr,
    "velum: wrote %s  samples=%zu (%.2f s @ %d Hz)  tokens=%zu  mel=%zu frames\n",
    a.out_wav.c_str(), r.audio.size(), (double)r.audio.size() / 24000.0,
    24000, r.tokens.size(), r.mel.size() / 80);
  return 0;
}
