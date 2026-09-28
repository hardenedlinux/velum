# Tokenizer data (committed)

These three files are the *runtime* data for the C++ `Qwen2Tokenizer`
(`src/llm/tokenizer.cpp`). They are committed so that building Velum needs no
Python and no access to the original model — `cmake --build build` copies them
into `build/tokenizer/` (the default `--tokenizer-dir`).

| file | content |
|---|---|
| `vocab.tsv` | Qwen2 base BPE vocabulary, `token<TAB>id` (151643 entries) |
| `merges.txt` | byte-pair merges, `a b` per line, in priority order (134935 entries) |
| `added_tokens.tsv` | CosyVoice3 special tokens, `id<TAB>content` (281 entries, ids 151643..151923) |

## Provenance & source verification

Generated from the CosyVoice3 text tokenizer (`CosyVoice-BlankEN`), which is the
Qwen2 tokenizer (`Qwen2Tokenizer`, Apache-2.0) plus CosyVoice3's special-token
list, via `tools/export_tokenizer.py` (standard library only).

| | value |
|---|---|
| HF model | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| HF revision | `29e01c4e8d000f4bcd70751be16fa94bf3d85a18` |
| CosyVoice commit | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` |
| Generated | 2026-09-28 |

The three *input* files (`CosyVoice-BlankEN/{vocab.json, merges.txt,
tokenizer_config.json}`) were verified to be byte-identical to the official
Hugging Face repository **and** to the ModelScope mirror before these outputs
were committed. Hugging Face stores these three files as regular git blobs, so
its authoritative content hash is the git blob SHA-1 (`oid`); ModelScope
publishes a SHA-256. Both match the local download:

| input file | bytes | git blob SHA-1 (HF `oid`) | SHA-256 |
|---|---|---|---|
| `vocab.json` | 2776833 | `4783fe10ac3adce15ac8f358ef5462739852c569` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | 1402109 | `90d3d82d027eadcc6a5e77c38eb82d43fc51b53b` | `ac8ff86a72bee70828fbc1119bc4398c6f3a9a6e490d7b0dbe917be025478bd0` |
| `tokenizer_config.json` | 1287 | `ff55d7b9eb1384e5d4d7e75dc0f564c1a8833d6e` | `482bd979881423375ca5414e4e0d94cd7c5349dbb17fffd46b4d36d71e62a1bc` |

`vocab.json` is byte-identical to `Qwen/Qwen2-0.5B`'s `vocab.json` (same git
`oid`). `merges.txt` is a CosyVoice-trimmed variant of the Qwen2 merge list
(134935 merges, 1402109 bytes, versus Qwen2-0.5B's 1671839 bytes).

## Output artifacts (committed)

| file | SHA-256 |
|---|---|
| `vocab.tsv` | `d22284131dad83f1fed301ea12fbfc98e2fbedc78299aee4ed82423b3885539f` |
| `merges.txt` | `ac8ff86a72bee70828fbc1119bc4398c6f3a9a6e490d7b0dbe917be025478bd0` |
| `added_tokens.tsv` | `0d0d5028c5291f5c279fad9eedc38c384db8d8769139eb5b439f89ede4182ab9` |

`merges.txt` is a verbatim copy of the source `merges.txt`, hence the identical
SHA-256. Regenerate with `python3 tools/export_tokenizer.py --out-dir
data/tokenizer` (see that script for the exact special-token list and id
assignment).

## License

Two upstream sources contribute to these files, both distributed under the
Apache License 2.0:

- **Qwen2** (`Qwen/Qwen2-0.5B`): `vocab.tsv` (the 151643-entry base vocabulary)
  is byte-identical to Qwen2-0.5B's `vocab.json`.
- **FunAudioLLM CosyVoice** (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`): the
  trimmed `merges.txt` (134935 merges) and the special-token list (hardcoded in
  `tools/export_tokenizer.py`, emitted to `added_tokens.tsv`) are CosyVoice's
  own additions, not Qwen2's.

The original Apache-2.0 text is reproduced in `LICENSE` in this directory. This
attribution is retained because these files are redistributed inside a GPLv3
project; Apache-2.0 is compatible with GPLv3.

On the CosyVoice license status: the files are taken from the Hugging Face
repository `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, which is tagged
`apache-2.0`; the model card's `Disclaimer` section additionally states that
"The content provided above is for academic purposes only and is intended to
demonstrate technical capabilities". The maintainers have not yet clarified
this tension (Hugging Face discussion #19, "Clarification on Commercial Use",
open at the time of writing). The above records the facts only and is not a
legal conclusion.

## TODO

Packaging/deployment: the runtime default `--tokenizer-dir` is `build/tokenizer`,
which only exists in a development build tree. When Velum is installed, add an
`install(FILES vocab.tsv merges.txt added_tokens.tsv DESTINATION
share/velum/tokenizer)` rule and make the default lookup fall back to that
installed path (a `build/` directory will not exist next to an installed binary).
