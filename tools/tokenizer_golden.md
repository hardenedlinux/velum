# tokenizer_golden.json

Golden reference for the velum C++ `Qwen2Tokenizer`: for each input text, the
exact sequence of token IDs that the official CosyVoice3 text tokenizer emits.
The C++ tokenizer must reproduce these IDs bit-exactly.

## How it was generated

The JSON was produced by encoding the sample texts with the *real*
`CosyVoice3Tokenizer.encode()` method (== `Qwen2TokenizerFast([text])["input_ids"]`,
over the `CosyVoice-BlankEN` base tokenizer), using the CosyVoice repository
only as a reference/ground-truth source:

```python
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

tok = get_qwen_tokenizer(token_path=".../Fun-CosyVoice3-0.5B/CosyVoice-BlankEN",
                         skip_special_tokens=False, version="cosyvoice3")
ids = tok.encode(text)   # list[int]
```

- CosyVoice commit: `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`
- Generated: 2026-09-28

The generation helper script is intentionally **not** committed; only this JSON
(and this note) are. The mapping is deterministic given the model, so the
contents can be reproduced by re-running the snippet above against the same
checkpoint.

## Cases covered

| # | Category | Example |
|---|----------|---------|
| 1 | Plain Chinese | `今天天气不错，我们一起去公园散步吧。` |
| 2 | Plain English | `Hello, how are you today?` |
| 3 | Mixed Chinese/English | `Hello world 你好世界，今天很开心。` |
| 4 | Pinyin pronunciation correction (PI) | `报道[j][ǐ]予好评` |
| 5 | CMU phoneme tokens | `The vowel [AA0] is in father, and [B] in boy.` |
| 6 | `<strong>` stress markers | `…<strong>勇气</strong>与<strong>智慧</strong>…` |
| 7 | Non-verbal markers | `…[laughter]…[breath]…[sigh]…` |
| 8 | Instruction prefix ending in `<|endofprompt|>` | `You are a helpful assistant. 请用普通话表达。<|endofprompt|>` |
