# DistilGPT2 LLM Estimate on Full Middlemarch

Input:

- File: `data/middlemarch_lowercase_letters_spaces.txt`
- Scope: full file
- Character set: ASCII lowercase letters `a-z` plus spaces only
- Model: `distilbert/distilgpt2`
- Method: sum `-log2 P(token_i | previous context)` over model tokens
- Interpretation: ideal arithmetic-coding length from this model's probabilities,
  excluding model weights, tokenizer files, headers, and finite-precision coding
  overhead.

LLM result:

| metric | value |
| --- | ---: |
| tokens scored | 363,977 |
| chars | 1,739,815 |
| bytes | 1,739,815 |
| total bits | 2,426,611.54 |
| bits/token | 6.6669 |
| bits/char | 1.3948 |
| bits/byte | 1.3948 |

Traditional compressors on the same full file:

| method | bytes | ratio | bits/byte | bits/char |
| --- | ---: | ---: | ---: | ---: |
| raw-utf8 | 1,739,815 | 1.0000 | 8.0000 | 8.0000 |
| gzip | 610,287 | 0.3508 | 2.8062 | 2.8062 |
| bz2 | 438,846 | 0.2522 | 2.0179 | 2.0179 |
| lzma | 480,184 | 0.2760 | 2.2080 | 2.2080 |

Compared with the 50,000-character prefix estimate (`1.4126 bit/char`), the
full-book estimate is slightly lower at `1.3948 bit/char`.
