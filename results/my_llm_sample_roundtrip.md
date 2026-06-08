# `.my-llm` Sample Round Trip

Input:

- File: `data/sample.txt`
- Model: `distilbert/distilgpt2`
- Device: Apple MPS
- Context window: `max_length=128`

Compression result:

| metric | value |
| --- | ---: |
| original bytes | 277 |
| tokenizer tokens | 55 |
| arithmetic payload bytes | 51 |
| total `.my-llm` bytes | 107 |
| bits/byte including header | 3.0903 |

Verification:

```bash
cmp -s data/sample.txt /private/tmp/sample.roundtrip.txt
```

Exit code: `0`, meaning the decoded file is byte-identical to the original.

This is a correctness smoke test for the MVP compressor. It is not intended as a
representative compression benchmark because the sample is very small and the
fixed `.my-llm` header dominates the file size.
