from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .arithmetic import ArithmeticDecoder, ArithmeticEncoder
from .my_llm_format import (
    HEADER_SIZE,
    MyLlmHeader,
    model_hash,
    pack_header,
    unpack_header,
)


DEFAULT_FREQ_BITS = 20


@dataclass(frozen=True)
class LlmCompressionResult:
    input_path: Path
    output_path: Path
    model: str
    device: str
    original_size: int
    token_count: int
    payload_size: int
    total_size: int
    bits_per_byte: float


@dataclass(frozen=True)
class LlmDecompressionResult:
    input_path: Path
    output_path: Path
    model: str
    original_size: int
    token_count: int


def default_compressed_path(path: Path) -> Path:
    return Path(str(path) + ".my-llm")


def compress_file(
    input_path: Path,
    output_path: Path | None,
    model_name: str,
    *,
    max_length: int = 1024,
    device: str | None = None,
    freq_bits: int = DEFAULT_FREQ_BITS,
    progress_every: int | None = 100,
) -> LlmCompressionResult:
    torch, tokenizer, model, resolved_device = _load_model(model_name, device)
    raw = input_path.read_bytes()
    text = raw.decode("utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokenizer_roundtrip = tokenizer.decode(
        token_ids,
        clean_up_tokenization_spaces=False,
    ).encode("utf-8")
    if tokenizer_roundtrip != raw:
        raise ValueError("tokenizer is not byte-exact for this UTF-8 input")
    if output_path is None:
        output_path = default_compressed_path(input_path)

    first_token_id = token_ids[0] if token_ids else 0
    header = MyLlmHeader(
        model_hash=model_hash(model_name),
        original_size=len(raw),
        token_count=len(token_ids),
        first_token_id=first_token_id,
    )

    if progress_every is not None:
        print(
            f"compressing {len(token_ids)} tokens on {resolved_device} "
            f"(max_length={max_length}, freq_bits={freq_bits})",
            file=sys.stderr,
            flush=True,
        )

    encoder = ArithmeticEncoder()
    with torch.no_grad():
        for index in range(1, len(token_ids)):
            cumulative = _next_token_cumulative(
                torch,
                tokenizer,
                model,
                resolved_device,
                token_ids[max(0, index - max_length) : index],
                freq_bits,
            )
            encoder.write(cumulative, token_ids[index])
            if progress_every is not None and (
                index == 1 or index % progress_every == 0 or index == len(token_ids) - 1
            ):
                print(
                    f"encoded {index}/{len(token_ids) - 1} predicted tokens",
                    file=sys.stderr,
                    flush=True,
                )

    body = encoder.finish() if len(token_ids) > 1 else b""
    output = pack_header(header) + body
    output_path.write_bytes(output)
    return LlmCompressionResult(
        input_path=input_path,
        output_path=output_path,
        model=model_name,
        device=resolved_device,
        original_size=len(raw),
        token_count=len(token_ids),
        payload_size=len(body),
        total_size=len(output),
        bits_per_byte=8.0 * len(output) / len(raw) if raw else 0.0,
    )


def decompress_file(
    input_path: Path,
    output_path: Path,
    model_name: str,
    *,
    max_length: int = 1024,
    device: str | None = None,
    freq_bits: int = DEFAULT_FREQ_BITS,
    progress_every: int | None = 100,
) -> LlmDecompressionResult:
    payload = input_path.read_bytes()
    header = unpack_header(payload)
    if header.model_hash != model_hash(model_name):
        raise ValueError("model name does not match the .my-llm header")

    if header.token_count == 0:
        output_path.write_bytes(b"")
        return LlmDecompressionResult(
            input_path=input_path,
            output_path=output_path,
            model=model_name,
            original_size=header.original_size,
            token_count=header.token_count,
        )

    torch, tokenizer, model, resolved_device = _load_model(model_name, device)
    token_ids = [header.first_token_id]
    decoder = ArithmeticDecoder(payload[HEADER_SIZE:])

    if progress_every is not None:
        print(
            f"decompressing {header.token_count} tokens on {resolved_device} "
            f"(max_length={max_length}, freq_bits={freq_bits})",
            file=sys.stderr,
            flush=True,
        )

    with torch.no_grad():
        while len(token_ids) < header.token_count:
            cumulative = _next_token_cumulative(
                torch,
                tokenizer,
                model,
                resolved_device,
                token_ids[-max_length:],
                freq_bits,
            )
            token_ids.append(decoder.read(cumulative))
            predicted = len(token_ids) - 1
            target = header.token_count - 1
            if progress_every is not None and (
                predicted == 1 or predicted % progress_every == 0 or predicted == target
            ):
                print(
                    f"decoded {predicted}/{target} predicted tokens",
                    file=sys.stderr,
                    flush=True,
                )

    text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    raw = text.encode("utf-8")
    if len(raw) != header.original_size:
        raise ValueError(
            "decoded byte length does not match header "
            f"({len(raw)} != {header.original_size})"
        )
    output_path.write_bytes(raw)
    return LlmDecompressionResult(
        input_path=input_path,
        output_path=output_path,
        model=model_name,
        original_size=header.original_size,
        token_count=header.token_count,
    )


def _load_model(model_name: str, device: str | None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LLM compression needs optional dependencies. Install with "
            "`pip install -e '.[llm]'` or use `uv run --extra llm ...`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _next_token_cumulative(
    torch,
    tokenizer,
    model,
    device: str,
    context_ids: list[int],
    freq_bits: int,
) -> list[int]:
    if not context_ids:
        raise ValueError("context_ids must not be empty")
    if freq_bits < 18:
        raise ValueError("freq_bits must be at least 18 for common LLM vocabularies")

    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    logits = model(input_ids).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1).detach().cpu()
    vocab_size = int(probs.numel())
    total_freq = 1 << freq_bits
    if total_freq <= vocab_size:
        raise ValueError("freq_bits is too small for the tokenizer vocabulary")

    scaled_mass = total_freq - vocab_size
    frequencies = torch.floor(probs * scaled_mass).to(torch.int64) + 1
    leftover = total_freq - int(frequencies.sum().item())
    if leftover > 0:
        frequencies[int(torch.argmax(probs).item())] += leftover

    cumulative = [0]
    running = 0
    for frequency in frequencies.tolist():
        running += int(frequency)
        cumulative.append(running)

    if cumulative[-1] != total_freq:
        raise ValueError("internal frequency quantization error")
    return cumulative
