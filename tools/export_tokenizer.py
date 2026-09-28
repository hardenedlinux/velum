#!/usr/bin/env python3
"""Export the CosyVoice3 text tokenizer's data for the C++ reimplementation.

This is a self-contained reimplementation of the original transformers-based
exporter: it uses only the Python 3 standard library and does not import
CosyVoice, ``transformers``, ``tokenizers``, ``torch``, ``tiktoken`` or
``whisper``. It reads the same base Qwen2 tokenizer files that
``CosyVoice3Tokenizer`` reads from the model directory and reproduces the
``add_special_tokens`` id assignment exactly, so the output is byte-for-byte
identical to the old script.

The C++ tokenizer needs three data files (plus the fixed GPT-2 byte alphabet,
which is computed in C++):

  - ``vocab.tsv``       : Qwen2 base BPE vocab (151643 entries, token-string -> id).
  - ``merges.txt``      : the 134935 byte-pair merges, one "a b" per line, in
                          priority order (rank = line index).
  - ``added_tokens.tsv``: the 281 special tokens (id<TAB>content), ids 151643..151923.

Only the standard library is required, so it can be run under a bare
``python3`` with no third-party packages and no ``PYTHONPATH`` pointing at
CosyVoice:

    python3 tools/export_tokenizer.py --out-dir build/tokenizer
"""

import argparse
import hashlib
import json
import os


# ---------------------------------------------------------------------------
# CosyVoice3 special tokens (hardcoded, verbatim and in order).
#
# Source: ~/Project/CosyVoice/cosyvoice/tokenizer/tokenizer.py, class
# ``CosyVoice3Tokenizer``. These are the ``eos_token`` / ``pad_token`` and the
# ``additional_special_tokens`` list that ``CosyVoice3Tokenizer.__init__``
# passes to ``self.tokenizer.add_special_tokens(...)``.
#
#   CosyVoice commit: 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
#   Extracted on:     2026-09-28
#
# This token list — and the trimmed ``merges.txt`` this script emits verbatim —
# are part of FunAudioLLM CosyVoice (Apache-2.0), not of the upstream Qwen2
# tokenizer. See data/tokenizer/README.md for full provenance and licensing.
#
# ``<|endoftext|>`` is used for both eos and pad. ``<|im_start|>`` and
# ``<|im_end|>`` are already present in the base tokenizer's
# ``tokenizer_config.json`` (``added_tokens_decoder``), so they are *not*
# re-assigned below - they keep their pre-existing ids. Every other token in
# ``ADDITIONAL_SPECIAL_TOKENS`` is new.
# ---------------------------------------------------------------------------
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|endoftext|>"

ADDITIONAL_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endofprompt|>",
    "[breath]",
    "<strong>",
    "</strong>",
    "[noise]",
    "[laughter]",
    "[cough]",
    "[clucking]",
    "[accent]",
    "[quick_breath]",
    "<laughter>",
    "</laughter>",
    "[hissing]",
    "[sigh]",
    "[vocalized-noise]",
    "[lipsmack]",
    "[mn]",
    "<|endofsystem|>",
    "[AA]",
    "[AA0]",
    "[AA1]",
    "[AA2]",
    "[AE]",
    "[AE0]",
    "[AE1]",
    "[AE2]",
    "[AH]",
    "[AH0]",
    "[AH1]",
    "[AH2]",
    "[AO]",
    "[AO0]",
    "[AO1]",
    "[AO2]",
    "[AW]",
    "[AW0]",
    "[AW1]",
    "[AW2]",
    "[AY]",
    "[AY0]",
    "[AY1]",
    "[AY2]",
    "[B]",
    "[CH]",
    "[D]",
    "[DH]",
    "[EH]",
    "[EH0]",
    "[EH1]",
    "[EH2]",
    "[ER]",
    "[ER0]",
    "[ER1]",
    "[ER2]",
    "[EY]",
    "[EY0]",
    "[EY1]",
    "[EY2]",
    "[F]",
    "[G]",
    "[HH]",
    "[IH]",
    "[IH0]",
    "[IH1]",
    "[IH2]",
    "[IY]",
    "[IY0]",
    "[IY1]",
    "[IY2]",
    "[JH]",
    "[K]",
    "[L]",
    "[M]",
    "[N]",
    "[NG]",
    "[OW]",
    "[OW0]",
    "[OW1]",
    "[OW2]",
    "[OY]",
    "[OY0]",
    "[OY1]",
    "[OY2]",
    "[P]",
    "[R]",
    "[S]",
    "[SH]",
    "[T]",
    "[TH]",
    "[UH]",
    "[UH0]",
    "[UH1]",
    "[UH2]",
    "[UW]",
    "[UW0]",
    "[UW1]",
    "[UW2]",
    "[V]",
    "[W]",
    "[Y]",
    "[Z]",
    "[ZH]",
    "[a]",
    "[ai]",
    "[an]",
    "[ang]",
    "[ao]",
    "[b]",
    "[c]",
    "[ch]",
    "[d]",
    "[e]",
    "[ei]",
    "[en]",
    "[eng]",
    "[f]",
    "[g]",
    "[h]",
    "[i]",
    "[ian]",
    "[in]",
    "[ing]",
    "[iu]",
    "[ià]",
    "[iàn]",
    "[iàng]",
    "[iào]",
    "[iá]",
    "[ián]",
    "[iáng]",
    "[iáo]",
    "[iè]",
    "[ié]",
    "[iòng]",
    "[ióng]",
    "[iù]",
    "[iú]",
    "[iā]",
    "[iān]",
    "[iāng]",
    "[iāo]",
    "[iē]",
    "[iě]",
    "[iōng]",
    "[iū]",
    "[iǎ]",
    "[iǎn]",
    "[iǎng]",
    "[iǎo]",
    "[iǒng]",
    "[iǔ]",
    "[j]",
    "[k]",
    "[l]",
    "[m]",
    "[n]",
    "[o]",
    "[ong]",
    "[ou]",
    "[p]",
    "[q]",
    "[r]",
    "[s]",
    "[sh]",
    "[t]",
    "[u]",
    "[uang]",
    "[ue]",
    "[un]",
    "[uo]",
    "[uà]",
    "[uài]",
    "[uàn]",
    "[uàng]",
    "[uá]",
    "[uái]",
    "[uán]",
    "[uáng]",
    "[uè]",
    "[ué]",
    "[uì]",
    "[uí]",
    "[uò]",
    "[uó]",
    "[uā]",
    "[uāi]",
    "[uān]",
    "[uāng]",
    "[uē]",
    "[uě]",
    "[uī]",
    "[uō]",
    "[uǎ]",
    "[uǎi]",
    "[uǎn]",
    "[uǎng]",
    "[uǐ]",
    "[uǒ]",
    "[vè]",
    "[w]",
    "[x]",
    "[y]",
    "[z]",
    "[zh]",
    "[à]",
    "[ài]",
    "[àn]",
    "[àng]",
    "[ào]",
    "[á]",
    "[ái]",
    "[án]",
    "[áng]",
    "[áo]",
    "[è]",
    "[èi]",
    "[èn]",
    "[èng]",
    "[èr]",
    "[é]",
    "[éi]",
    "[én]",
    "[éng]",
    "[ér]",
    "[ì]",
    "[ìn]",
    "[ìng]",
    "[í]",
    "[ín]",
    "[íng]",
    "[ò]",
    "[òng]",
    "[òu]",
    "[ó]",
    "[óng]",
    "[óu]",
    "[ù]",
    "[ùn]",
    "[ú]",
    "[ún]",
    "[ā]",
    "[āi]",
    "[ān]",
    "[āng]",
    "[āo]",
    "[ē]",
    "[ēi]",
    "[ēn]",
    "[ēng]",
    "[ě]",
    "[ěi]",
    "[ěn]",
    "[ěng]",
    "[ěr]",
    "[ī]",
    "[īn]",
    "[īng]",
    "[ō]",
    "[ōng]",
    "[ōu]",
    "[ū]",
    "[ūn]",
    "[ǎ]",
    "[ǎi]",
    "[ǎn]",
    "[ǎng]",
    "[ǎo]",
    "[ǐ]",
    "[ǐn]",
    "[ǐng]",
    "[ǒ]",
    "[ǒng]",
    "[ǒu]",
    "[ǔ]",
    "[ǔn]",
    "[ǘ]",
    "[ǚ]",
    "[ǜ]",
]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir",
                    default=os.path.expanduser(
                        "~/Project/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"))
    ap.add_argument("--out-dir", default="build/tokenizer")
    args = ap.parse_args()

    token_path = os.path.join(args.model_dir, "CosyVoice-BlankEN")

    # 1. Base BPE vocab (token-string -> id) from vocab.json. This is exactly
    #    the ``model.vocab`` dict of the serialized backend tokenizer.
    vocab = _read_json(os.path.join(token_path, "vocab.json"))

    # 2. Byte-pair merges from merges.txt, in priority order (rank = line index).
    #    Mirrors how the tokenizers library loads merges.txt: skip blank lines
    #    and any "#version" header, split each remaining line into its two
    #    tokens.
    merges = []
    with open(os.path.join(token_path, "merges.txt"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("#version"):
                continue
            a, b = line.split()
            merges.append((a, b))

    # 3. Pre-existing added tokens from tokenizer_config.json. This model ships
    #    no ``added_tokens.json``; the three Qwen2 chat tokens are declared via
    #    ``added_tokens_decoder``. They keep their fixed ids.
    tok_cfg = _read_json(os.path.join(token_path, "tokenizer_config.json"))
    added_decoder = tok_cfg.get("added_tokens_decoder", {})
    pre_existing = sorted((int(i), info["content"])
                          for i, info in added_decoder.items())

    # Reproduce HuggingFace ``add_special_tokens`` id assignment: the special
    # tokens are tried in order [eos, pad] + additional_special_tokens; a token
    # already in the vocabulary keeps its id, a new token is assigned the next
    # free id starting from len(base_vocab) + len(pre_existing).
    present = set(vocab) | {content for _, content in pre_existing}
    next_id = len(vocab) + len(pre_existing)
    new_added = []
    for tok in [EOS_TOKEN, PAD_TOKEN] + ADDITIONAL_SPECIAL_TOKENS:
        if tok not in present:
            new_added.append((next_id, tok))
            present.add(tok)
            next_id += 1

    added = sorted(pre_existing + new_added, key=lambda x: x[0])

    os.makedirs(args.out_dir, exist_ok=True)

    # vocab.tsv — "token<TAB>id" for the 151643 base BPE tokens. Byte-level
    # tokens never contain TAB/newline (bytes are mapped into the GPT-2 byte
    # alphabet), so TAB is an unambiguous separator and the C++ side can parse
    # it with plain string ops (no JSON parser needed).
    with open(os.path.join(args.out_dir, "vocab.tsv"), "w", encoding="utf-8") as f:
        for token, vid in sorted(vocab.items(), key=lambda kv: kv[1]):
            assert "\t" not in token and "\n" not in token, repr(token)
            f.write(f"{token}\t{vid}\n")

    # merges.txt — byte-pair merges in priority order (rank = line index).
    with open(os.path.join(args.out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")

    # added_tokens.tsv — the special tokens with their fixed ids.
    with open(os.path.join(args.out_dir, "added_tokens.tsv"), "w", encoding="utf-8") as f:
        for vid, content in added:
            f.write(f"{vid}\t{content}\n")

    print(f"vocab:        {len(vocab)} entries")
    print(f"merges:       {len(merges)} entries")
    print(f"added_tokens: {len(added)} entries (ids {added[0][0]}..{added[-1][0]})")
    print(f"wrote -> {os.path.abspath(args.out_dir)}")

    for name in ("vocab.tsv", "merges.txt", "added_tokens.tsv"):
        path = os.path.join(args.out_dir, name)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        print(f"sha256 {name}: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
