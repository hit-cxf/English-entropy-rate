# Qwen3-0.6B LLM Estimate on Middlemarch 50k

Input:

- File: `data/middlemarch_lowercase_letters_spaces.txt`
- Prefix: first 50,000 characters
- Character set: ASCII lowercase letters `a-z` plus spaces only
- Model: `Qwen/Qwen3-0.6B-Base`
- Device: Apple MPS
- Context window: `max_length=1024`, `stride=512`
- Method: sum `-log2 P(token_i | previous context)` over model tokens
- Interpretation: ideal arithmetic-coding length from this model's probabilities,
  excluding model weights, tokenizer files, headers, and finite-precision coding
  overhead.

LLM result:

| metric | value |
| --- | ---: |
| tokens scored | 10,325 |
| chars | 50,000 |
| bytes | 50,000 |
| total bits | 57,364.40 |
| bits/token | 5.5559 |
| bits/char | 1.1473 |
| bits/byte | 1.1473 |

Comparison on the same 50,000-character prefix:

| method/model | bits/char |
| --- | ---: |
| raw-utf8 | 8.0000 |
| gzip | 3.0640 |
| bz2 | 2.6194 |
| lzma | 2.8755 |
| `distilbert/distilgpt2` | 1.4126 |
| `Qwen/Qwen3-0.6B-Base` | 1.1473 |

Qwen3 uses a different tokenizer than GPT-2, so `bits/token` is not directly
comparable across models. `bits/char` is the model-comparable metric here.
