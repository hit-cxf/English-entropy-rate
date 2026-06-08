# Middlemarch Lowercase Letters/Spaces Baseline

Input:

- File: `data/middlemarch_lowercase_letters_spaces.txt`
- Derived from: `data/middlemarch_gutenberg_145.txt`
- Transform: lowercase all letters, replace every non-`a-z` character with a
  space boundary, and collapse consecutive spaces.
- Raw bytes: 1,739,815

Validation:

- Character set: ASCII lowercase letters `a-z` plus spaces only
- Consecutive spaces: none

Results:

| method | bytes | ratio | bits/byte | bits/char |
| --- | ---: | ---: | ---: | ---: |
| raw-utf8 | 1,739,815 | 1.0000 | 8.0000 | 8.0000 |
| gzip | 610,287 | 0.3508 | 2.8062 | 2.8062 |
| bz2 | 438,846 | 0.2522 | 2.0179 | 2.0179 |
| lzma | 480,184 | 0.2760 | 2.2080 | 2.2080 |
