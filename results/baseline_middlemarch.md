# Middlemarch Baseline

Input:

- File: `data/middlemarch_gutenberg_145.txt`
- Source: Project Gutenberg eBook #145, Plain Text UTF-8
- Raw bytes: 1,865,684

Results:

| method | bytes | ratio | bits/byte | bits/char |
| --- | ---: | ---: | ---: | ---: |
| raw-utf8 | 1,865,684 | 1.0000 | 8.0000 | 8.1423 |
| gzip | 701,571 | 0.3760 | 3.0083 | 3.0618 |
| bz2 | 505,013 | 0.2707 | 2.1655 | 2.2040 |
| lzma | 553,220 | 0.2965 | 2.3722 | 2.4144 |

The text contains non-ASCII punctuation, so `bits/char` is slightly larger than
`bits/byte` for the raw UTF-8 baseline.
