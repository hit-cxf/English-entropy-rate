# English Entropy Rate

Small experiments for estimating English text entropy rate from compression and
language-model prediction.

The project has two complementary estimators:

- `baseline`: compare ordinary lossless compressors against raw UTF-8 text.
- `llm`: estimate the ideal arithmetic-coding length from a language model's
  next-token probabilities.

The LLM estimator does not write a compressed file yet. It computes the number
of bits an ideal arithmetic coder would need if it used the same model
probabilities.

## Quick Start

Run the traditional compression baselines:

```bash
PYTHONPATH=src python -m english_entropy_rate baseline data/sample.txt
```

Run the LLM ideal-code estimator after installing optional dependencies:

```bash
PYTHONPATH=src python -m english_entropy_rate llm data/sample.txt --model distilbert/distilgpt2
```

For a large file, start with a prefix:

```bash
PYTHONPATH=src python -m english_entropy_rate llm \
  data/middlemarch_lowercase_letters_spaces.txt \
  --model distilbert/distilgpt2 \
  --limit-chars 50000 \
  --max-length 1024 \
  --stride 512
```

On Apple Silicon, the CLI automatically prefers the MPS backend when available.
Override it with `--device cpu` or `--device mps` when comparing hardware.

For larger texts, use a smaller model first (`distilgpt2`) and increase
`--stride` only after the workflow is behaving as expected.

## Metrics

For a text file with `N` UTF-8 bytes and `C` Python characters:

```text
bits_per_byte = compressed_bits / N
bits_per_char = compressed_bits / C
```

For the LLM estimator:

```text
ideal_bits = sum_i -log2 P(token_i | previous context)
```

This is the compressed length an ideal arithmetic coder would approach,
excluding file headers, model parameters, tokenizer metadata, and finite-precision
coding overhead.

## Notes

- Use held-out text that the model probably has not memorized.
- Report the symbol unit clearly: byte, character, token, or letter.
- A pretrained model is shared side information. If you count model weights as
  part of the compressed payload, short files become extremely expensive.
