#!/usr/bin/env python3
"""End-to-end segmented-input check: `velum --segments-file`.

Runs the full C++ CLI on a 3-segment `--segments-file` JSON array with a
distinct instruction per segment (the per-sentence instruct path) and verifies:

  1. the concatenated output wav exists and is non-empty;
  2. each per-segment wav (`--segments-dir`/segment_001..003.wav) exists;
  3. the concatenated duration is exactly the sum of the per-segment durations
     plus the inter-segment gaps (i.e. duration >= sum + gaps, which is the
     acceptance criterion for segmented concatenation).

It also checks that the instruction *wrapper* rejects a caller-supplied prefix
or `<|endofprompt|>` with a nonzero exit (Velum adds both itself).

Needs the `models` target (llm/flow/hift gguf) and the build-tree assets
(tokenizer, prompt bundle, RNG buffers) — the same prerequisites as the other
numerical ctests. This script is stdlib-only (json/os/subprocess/wave) so it
runs under the repo's `.venv` python without extra deps.

Usage (via ctest, see CMakeLists.txt):
    VELUM_BIN=build/velum VELUM_BUILD_DIR=build \
        .venv/bin/python tests/verify_segments.py
"""

import json
import os
import subprocess
import sys
import tempfile
import wave

ROOT = os.environ.get(
    "VELUM_SOURCE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.environ.get("VELUM_BUILD_DIR", os.path.join(ROOT, "build"))
BIN = os.environ.get("VELUM_BIN", os.path.join(BUILD, "velum"))

SAMPLE_RATE = 24000
GAP_MS = 200
GAP_SAMPLES = GAP_MS * SAMPLE_RATE // 1000  # 4800

# Three segments. The per-segment "instruct" values are instruction *bodies*
# drawn from the upstream presets (cosyvoice/utils/common.py instruct_list);
# Velum adds the "You are a helpful assistant. " prefix and <|endofprompt|>.
# The third segment omits "instruct" to exercise the fallback to --instruct.
SEGMENTS = [
    {"text": "欢迎来到我们的节目。", "instruct": "请用广东话表达。"},
    {"text": "今天我们聊聊天气情况。", "instruct": "请非常开心地说一句话。"},
    {"text": "感谢收听，我们下次再见。"},
]

# Fallback instruction body for segments without their own "instruct"
# (Velum's default instruction body).
DEFAULT_INSTRUCT = "请用普通话表达。"


def wav_frames(path):
    with wave.open(path, "rb") as w:
        return w.getnframes()


def run_expect_fail(args, needle):
    """Run velum expecting a nonzero exit and `needle` in stderr."""
    proc = subprocess.run([BIN] + args, capture_output=True, text=True)
    assert proc.returncode != 0, (
        f"expected nonzero exit for {args!r}, got {proc.returncode}")
    assert needle in proc.stderr, (
        f"expected {needle!r} in stderr, got: {proc.stderr}")
    print(f"ok: rejected with nonzero exit ({needle!r})", flush=True)


def main():
    llm = os.path.join(ROOT, "models", "llm.gguf")
    flow = os.path.join(ROOT, "models", "flow.gguf")
    hift = os.path.join(ROOT, "models", "hift.gguf")
    for p in (BIN, llm, flow, hift):
        if not os.path.exists(p):
            sys.exit(f"{p} not found; run `cmake --build build --target models` first")

    hift_source = os.path.join(BUILD, "hift_source.bin")
    flow_noise = os.path.join(BUILD, "flow_noise.bin")
    tokenizer_dir = os.path.join(BUILD, "tokenizer")
    prompt_dir = os.path.join(BUILD, "prompt")

    tmp = tempfile.mkdtemp(prefix="velum_segments_")
    segments_json = os.path.join(tmp, "segments.json")
    seg_dir = os.path.join(tmp, "segments")
    out_wav = os.path.join(tmp, "combined.wav")

    with open(segments_json, "w", encoding="utf-8") as f:
        json.dump(SEGMENTS, f, ensure_ascii=False)

    cmd = [BIN,
           "--segments-file", segments_json,
           "--segments-dir", seg_dir,
           "--out", out_wav,
           "--segment-gap-ms", str(GAP_MS),
           "--instruct", DEFAULT_INSTRUCT,
           "--seed", "0",
           "--llm", llm, "--flow", flow, "--hift", hift,
           "--hift-source", hift_source,
           "--flow-noise", flow_noise,
           "--tokenizer-dir", tokenizer_dir,
           "--prompt-dir", prompt_dir]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # 1. concatenated output exists and is non-empty.
    assert os.path.exists(out_wav), f"missing output {out_wav}"
    combined = wav_frames(out_wav)
    assert combined > 0, "combined wav is empty"

    # 2. each per-segment wav exists.
    per_seg = []
    for i in range(1, len(SEGMENTS) + 1):
        p = os.path.join(seg_dir, f"segment_{i:03d}.wav")
        assert os.path.exists(p), f"missing per-segment wav {p}"
        per_seg.append(wav_frames(p))

    # 3. concatenated duration == sum(per-segment) + (n-1) gaps.
    expected = sum(per_seg) + (len(SEGMENTS) - 1) * GAP_SAMPLES
    print(f"segments={per_seg} gaps={GAP_SAMPLES} sum={sum(per_seg)} "
          f"expected={expected} combined={combined}", flush=True)
    assert combined >= expected, \
        f"combined {combined} < sum+gaps {expected}"
    assert combined == expected, \
        f"combined {combined} != exact sum+gaps {expected} (unexpected overlap/crossfade)"

    # 4. the instruction wrapper rejects a caller-supplied prefix / <|endofprompt|>.
    base = ["--prompt-dir", prompt_dir, "--out", os.path.join(tmp, "err.wav")]
    run_expect_fail(
        ["--text", "你好。",
         "--instruct", "You are a helpful assistant. 请用普通话表达。"] + base,
        'already contains the "You are a helpful assistant" prefix')
    run_expect_fail(
        ["--text", "你好。",
         "--instruct", "请用普通话表达。<|endofprompt|>"] + base,
        "already contains <|endofprompt|>")

    print("PASS: 3-segment multi-instruct synthesis concatenates to "
          "sum(segments) + gaps with per-segment wavs written, and the "
          "instruction wrapper rejects a caller-supplied prefix/<|endofprompt|>.")


if __name__ == "__main__":
    main()
