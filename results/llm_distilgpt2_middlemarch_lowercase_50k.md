# DistilGPT2 LLM Estimate on Middlemarch 50k

Input:

- File: `data/middlemarch_lowercase_letters_spaces.txt`
- Prefix: first 50,000 characters
- Character set: ASCII lowercase letters `a-z` plus spaces only
- Model: `distilbert/distilgpt2`
- Method: sum `-log2 P(token_i | previous context)` over model tokens
- Interpretation: ideal arithmetic-coding length from this model's probabilities,
  excluding model weights, tokenizer files, headers, and finite-precision coding
  overhead.

LLM result:

| metric | value |
| --- | ---: |
| tokens scored | 10,537 |
| chars | 50,000 |
| bytes | 50,000 |
| total bits | 70,629.19 |
| bits/token | 6.7030 |
| bits/char | 1.4126 |
| bits/byte | 1.4126 |

Traditional compressors on the same 50,000-character prefix:

| method | bytes | ratio | bits/byte | bits/char |
| --- | ---: | ---: | ---: | ---: |
| raw-utf8 | 50,000 | 1.0000 | 8.0000 | 8.0000 |
| gzip | 19,150 | 0.3830 | 3.0640 | 3.0640 |
| bz2 | 16,371 | 0.3274 | 2.6194 | 2.6194 |
| lzma | 17,972 | 0.3594 | 2.8755 | 2.8755 |

The LLM number is not an actual compressed file size yet. It is the theoretical
payload length an arithmetic coder would approach if both encoder and decoder
shared exactly this model and tokenizer.
