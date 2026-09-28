# Annotation syntax (text-side control tokens)

> **Stability guarantee.** The annotation syntax described here is a *public,
> stable* interface. Any incompatible change to a token's spelling, the token
> set, or how annotations compose — anything that would change the token IDs a
> given input text produces — requires advance notice and a **major** version
> bump. Bug-fix releases must not alter tokenization.

## What this document covers

Velum passes the input text to the tokenizer **verbatim**. It performs no text
normalization, no sentence splitting, and no Hanzi → pinyin conversion. The only
"language" the tokenizer adds on top of the base ByteLevel-BPE vocabulary is a
fixed set of *special tokens* (token IDs `151643`–`151923`), committed in
`data/tokenizer/added_tokens.tsv`. This document is the reference for writing
those tokens in the text you feed to `velum` (via `--text`, `--text-file`, or
the per-segment `"text"` field of `--segments-file`).

Annotations are matched as **whole tokens** (longest-match wins, e.g. `[ing]`
over `[in]` over `[i]`). Bracketed or angle-bracketed strings that are **not** in
the token set are *not* special — they fall back to ordinary subword
tokenization.

The four families, by token-ID range:

| Family | IDs | Count | Purpose |
|---|---|---|---|
| Control | `151643`–`151646`, `151663` | 5 | instruction framing (`<|endofprompt|>`, chat tags) |
| Markers | `151647`–`151662` | 16 | paralinguistic / event control (`[breath]`, `<strong>`, …) |
| CMU phonemes | `151664`–`151747` | 84 | explicit English pronunciation |
| Pinyin | `151748`–`151923` | 176 | explicit Mandarin pronunciation |

---

## 1. Instruction format (control tokens)

Velum's instruction-following mode uses the CosyVoice3 `instruct2` prompt
format:

```
You are a helpful assistant. <instruction><|endofprompt|>
```

- The literal prefix `You are a helpful assistant. ` (with trailing space) and
  the trailing `<|endofprompt|>` (token `151646`) are **added automatically** by
  Velum. The caller writes only the *instruction body* — via `--instruct`, or the
  per-segment `"instruct"` field of `--segments-file` — e.g.
  `--instruct "请用广东话表达。"`.
- Supplying the prefix or `<|endofprompt|>` yourself is an **error**: Velum
  refuses (nonzero exit) rather than silently de-duplicating.
- `<|endofprompt|>` terminates the instruction and is required; upstream asserts
  its presence (`cosyvoice/llm/llm.py` raises "`<|endofprompt|>` not detected"
  when it is missing).
- The default instruction body is `请用普通话表达。`.

### 1.1 Officially trained instruction bodies

The upstream presets (`cosyvoice/utils/common.py` `instruct_list`) are the 26
instruction bodies the model was trained with. Velum wraps each verbatim as
above. Any body *not* in this list (including the default `请用普通话表达。`)
is **usable — observed effective in testing — but not officially trained**;
validate it against your specific copy before production use.

**Dialect / language (17):**
`请用广东话表达。`, `请用东北话表达。`, `请用甘肃话表达。`, `请用贵州话表达。`,
`请用河南话表达。`, `请用湖北话表达。`, `请用湖南话表达。`, `请用江西话表达。`,
`请用闽南话表达。`, `请用宁夏话表达。`, `请用山西话表达。`, `请用陕西话表达。`,
`请用山东话表达。`, `请用上海话表达。`, `请用四川话表达。`, `请用天津话表达。`,
`请用云南话表达。`

**Voice / volume (2):**
`Please say a sentence as loudly as possible.`,
`Please say a sentence in a very soft voice.`

**Speed (2):**
`请用尽可能慢地语速说一句话。`, `请用尽可能快地语速说一句话。`

**Emotion (3):**
`请非常开心地说一句话。`, `请非常伤心地说一句话。`, `请非常生气地说一句话。`

**Style (2):**
`我想体验一下小猪佩奇风格，可以吗？`, `你可以尝试用机器人的方式解答吗？`

The remaining control tokens are chat-framing tags that are part of the token
set but not used by Velum's `instruct2` path: `<|endoftext|>` (`151643`),
`<|im_start|>` (`151644`), `<|im_end|>` (`151645`), `<|endofsystem|>` (`151663`).

> Note: CosyVoice **1**'s cross-lingual tags `<|zh|>`, `<|en|>`, `<|ja|>`,
> `<|yue|>`, `<|ko|>` are **not** present in the CosyVoice3 tokenizer (they are
> not in the base vocab nor in `added_tokens.tsv`). Do not use them with Velum.

---

## 2. Non-language markers (paralinguistic / event control)

These are written literally into the text and control prosody or non-speech
events. Single-token markers use `[...]`; the two emphasis/laughter controls use
paired `<…>` / `</…>` tags.

| Token | ID | Meaning |
|---|---|---|
| `[breath]` | 151647 | audible breath |
| `<strong>` / `</strong>` | 151648 / 151649 | emphasis span (see §5) |
| `[noise]` | 151650 | background noise |
| `[laughter]` | 151651 | laughter |
| `[cough]` | 151652 | cough |
| `[clucking]` | 151653 | tongue click ("tsk") |
| `[accent]` | 151654 | accented/foreign pronunciation |
| `[quick_breath]` | 151655 | short intake of breath |
| `<laughter>` / `</laughter>` | 151656 / 151657 | laughter span |
| `[hissing]` | 151658 | hissing sound |
| `[sigh]` | 151659 | sigh |
| `[vocalized-noise]` | 151660 | non-verbal vocalization |
| `[lipsmack]` | 151661 | lip smack |
| `[mn]` | 151662 | filled pause ("mm", "嗯") |

---

## 3. Pinyin

Explicit Mandarin pronunciation is written as **bracket tokens**. A syllable is
one `[initial]` token followed by one `[final]` token, where the final carries
the tone as a diacritic. There is no automatic Hanzi → pinyin conversion: the
caller spells out the pronunciation (this is CosyVoice3's "hotfix" /
pronunciation-inpainting feature).

```
报道[j][ǐ]予好评      # 报道 jǐ 予好评 — [j] = initial, [ǐ] = final (3rd tone)
```

### 3.1 Initials (23)

`[b] [p] [m] [f] [d] [t] [n] [l] [g] [k] [h] [j] [q] [x] [zh] [ch] [sh] [r]
[z] [c] [s] [w] [y]`

### 3.2 Finals — bare (22, neutral tone)

`[a] [ai] [an] [ang] [ao] [e] [ei] [en] [eng] [i] [ian] [in] [ing] [iu] [o]
[ong] [ou] [u] [uang] [ue] [un] [uo]`

A bare final (no tone diacritic) is the neutral tone (轻声), e.g. the sentence
particle 吗 in `好[h][ǎo][m][a]` → `[m][a]`.

### 3.3 Finals — toned

A toned final is the bare final with the tone diacritic on the vowel, e.g.
`[ǐ]` (= `[i]`, 3rd tone), `[ōng]` (= `[ong]`, 1st tone), `[àn]` (= `[an]`,
4th tone). The four marked tones:

| Tone | Name | Diacritic | Example |
|---|---|---|---|
| 1 | 阴平 | macron | `[ā] [ē] [ī] [ō] [ū]` |
| 2 | 阳平 | acute | `[á] [é] [í] [ó] [ú]` |
| 3 | 上声 | caron | `[ǎ] [ě] [ǐ] [ǒ] [ǔ]` |
| 4 | 去声 | grave | `[à] [è] [ì] [ò] [ù]` |

The complete enumeration of the 176 pinyin tokens (initials + bare + toned
finals) is the authoritative list in `data/tokenizer/added_tokens.tsv`
(`151748`–`151923`). Note two quirks of the set: `ü` is written `v` in the
`üe` final (`[vè]`) but `ǘ/ǚ/ǜ` standalone, and the `er` final appears only in
its toned forms (`[èr] [ér] [ěr]`).

### 3.4 Zero-initial syllables

A syllable whose pinyin starts with a vowel (安 `ān`, 爱 `ài`, 欧 `ōu`, 二 `èr`)
has no initial and is written as the toned final alone:

```
[ān]    # 安
[ài]    # 爱
[ōu]    # 欧
[èr]    # 二
```

### 3.5 Composition rules

- Syllable = `[initial]` `[final]`, e.g. `[n][ǐ]`, `[zh][ōng]`, `[sh][àng]`.
- Zero-initial syllable = `[final]` alone.
- Neutral-tone syllable = `[initial]` `[bare final]` (e.g. `[m][a]`) or
  `[bare final]` alone.
- Multi-syllable words are written syllable-by-syllable:
  `[n][ǐ][h][ǎo]` (你好), `[zh][ōng][w][én]` (中文), `[b][a][b][a]` (爸爸).

---

## 4. CMU phonemes (explicit English pronunciation)

English pronunciation is written as `[PHONEME]` tokens using the CMU (ARPAbet)
symbol set. Vowels carry an optional stress digit; consonants do not.

- **Vowels** — 15 symbols, each with four forms: bare, `0` (unstressed),
  `1` (primary stress), `2` (secondary stress):
  `AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW`
  (e.g. `[AA]`, `[AA0]`, `[AA1]`, `[AA2]`).
- **Consonants** — 24 symbols, no stress suffix:
  `B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH`
  (e.g. `[B]`, `[ZH]`, `[NG]`).

A word is spelled phoneme-by-phoneme, e.g. `[HH][EH1][L][OW0]` ("hello") or
`[AA0]` for the first vowel of "father".

---

## 5. `<strong>` emphasis

`<strong>` (`151648`) and `</strong>` (`151649`) mark an emphasis span:

```
在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。
```

They are two opaque tokens passed through to the LLM. The tokenizer and frontend
perform **no pairing or nesting validation**. Consequently:

- **Nesting** (`<strong>a<strong>b</strong>c</strong>`) has **no defined
  behavior** — do not use it.
- **Cross-punctuation spans** (a `<strong>` whose `</strong>` is separated by a
  sentence boundary) have **no defined behavior** — do not rely on it.

---

## Verification

Every rule above is pinned by a golden case in `tools/tokenizer_golden.json`,
checked bit-exactly against the tokenizer by `tests/verify_tokenizer.py`
(`cmake --build build && .venv/bin/python tests/verify_tokenizer.py`):

| Rule | Golden case text | Asserts |
|---|---|---|
| instruct format | `You are a helpful assistant. 请用普通话表达。<|endofprompt|>` | prefix + `<|endofprompt|>` → `…, 151646` |
| pinyin composition | `[n][ǐ][h][ǎo]`, `[zh][ōng][w][én]`, `[sh][àng][h][ǎi]` | initial + toned final per syllable |
| zero-initial | `[ān]`, `[ài]`, `[ōu]`, `[èr]` | single toned-final token |
| bare final (neutral) | `[m][a]`, `[b][a][b][a]` | bare final token without tone |
| CMU phonemes | `[AA]`, `[AA2]`, `[HH][EH1][L][OW0]` | vowel bare/stress + consonants |
| markers (paired) | `<laughter>哈哈</laughter>` | `<laughter>`/`</laughter>` framing |
| markers (single) | `[noise][cough][clucking][accent][quick_breath][hissing][vocalized-noise][lipsmack][mn]` | each single-token marker |
| emphasis | `在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。` | `<strong>`/`</strong>` tokens |
| single markers | `他[laughter]笑了笑[breath]，然后[sigh]叹了口气。` | `[laughter]`/`[breath]`/`[sigh]` |
