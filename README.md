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

Compress a small UTF-8 text file to the project format:

```bash
PYTHONPATH=src python -m english_entropy_rate compress \
  data/sample.txt \
  --model distilbert/distilgpt2
```

The default output path appends `.my-llm`, for example
`data/sample.txt.my-llm`. Decompress with the same model:

```bash
PYTHONPATH=src python -m english_entropy_rate decompress \
  data/sample.txt.my-llm \
  /tmp/sample.roundtrip.txt \
  --model distilbert/distilgpt2
```

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

## `.my-llm` Format

The compressor writes a fixed-size binary header followed by an arithmetic-coded
payload:

```text
8 bytes   magic/version: MYLLM001
32 bytes  sha256(model_name)
8 bytes   original UTF-8 byte length, big-endian unsigned integer
4 bytes   tokenizer token count, big-endian unsigned integer
4 bytes   first token id, big-endian unsigned integer
...       arithmetic-coded predicted tokens
```

The first token is stored directly because the current MVP scores
`P(token_i | previous tokens)`. The arithmetic payload stores tokens
`1..N-1`. The byte length and token count tell the decoder exactly when to stop;
without that metadata, arithmetic-coded intervals for different message lengths
would be ambiguous.

This is a correctness-first implementation. It recomputes the model context for
each predicted token, so it is suitable for small files and prefixes. A future
version should add KV-cache decoding and batched encoder-side scoring.
