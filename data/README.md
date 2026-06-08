# Data

## `middlemarch_gutenberg_145.txt`

- Title: *Middlemarch*
- Author: George Eliot
- Source: Project Gutenberg eBook #145
- URL: https://www.gutenberg.org/ebooks/145.txt.utf-8
- Format: Plain Text UTF-8

The file includes the Project Gutenberg header and license text. For some
experiments we may later add a cleaned version that keeps only the novel body.

## `middlemarch_lowercase_letters_spaces.txt`

Derived from `middlemarch_gutenberg_145.txt` with:

```bash
PYTHONPATH=src python3 -m english_entropy_rate clean \
  data/middlemarch_gutenberg_145.txt \
  data/middlemarch_lowercase_letters_spaces.txt
```

The file contains only ASCII lowercase letters `a-z` and spaces. Uppercase
letters are lowercased, and every other character is treated as a word boundary.
