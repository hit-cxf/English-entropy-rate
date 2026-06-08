from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LlmEntropyResult:
    model: str
    device: str
    max_length: int
    stride: int
    tokens_scored: int
    chars: int
    bytes: int
    total_bits: float
    bits_per_token: float
    bits_per_char: float
    bits_per_byte: float


def estimate_llm_bits(
    path: Path,
    model_name: str,
    *,
    limit_chars: int | None = None,
    max_length: int | None = None,
    stride: int = 512,
    device: str | None = None,
    progress_every: int | None = None,
) -> LlmEntropyResult:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LLM estimation needs optional dependencies. Install with "
            "`pip install -e '.[llm]'` or use `uv run --extra llm ...`."
        ) from exc

    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if limit_chars is not None:
        if limit_chars <= 0:
            raise ValueError("limit_chars must be positive")
        text = text[:limit_chars]
        raw = text.encode("utf-8")
    if not text:
        raise ValueError("input text must not be empty")

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

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    sequence_length = input_ids.size(1)
    if sequence_length < 2:
        raise ValueError("input text is too short to score next-token prediction")

    if max_length is None:
        max_length = int(
            getattr(
                model.config,
                "n_positions",
                getattr(model.config, "max_position_embeddings", 1024),
            )
        )
    if stride <= 0 or stride > max_length:
        raise ValueError("stride must be in the range 1..max_length")
    if progress_every is not None and progress_every <= 0:
        raise ValueError("progress_every must be positive")

    total_nll = 0.0
    tokens_scored = 0
    total_windows = math.ceil(sequence_length / stride)

    if progress_every is not None:
        print(
            "scoring "
            f"{sequence_length} tokens on {device} "
            f"(max_length={max_length}, stride={stride}, windows~{total_windows})",
            file=sys.stderr,
            flush=True,
        )

    with torch.no_grad():
        previous_end = 0
        for window_index, begin in enumerate(range(0, sequence_length, stride), start=1):
            end = min(begin + max_length, sequence_length)
            target_length = end - previous_end
            window = input_ids[:, begin:end]
            labels = window.clone()
            labels[:, :-target_length] = -100

            outputs = model(window, labels=labels)
            scored = int((labels[:, 1:] != -100).sum().item())
            if scored:
                total_nll += float(outputs.loss.item()) * scored
                tokens_scored += scored

            previous_end = end
            if progress_every is not None and (
                window_index == 1
                or window_index % progress_every == 0
                or end == sequence_length
            ):
                print(
                    f"window {window_index}/{total_windows}: "
                    f"scored {tokens_scored}/{sequence_length - 1} tokens",
                    file=sys.stderr,
                    flush=True,
                )
            if end == sequence_length:
                break

    if tokens_scored == 0:
        raise ValueError("no tokens were scored")

    total_bits = total_nll / math.log(2)
    return LlmEntropyResult(
        model=model_name,
        device=device,
        max_length=max_length,
        stride=stride,
        tokens_scored=tokens_scored,
        chars=len(text),
        bytes=len(raw),
        total_bits=total_bits,
        bits_per_token=total_bits / tokens_scored,
        bits_per_char=total_bits / len(text),
        bits_per_byte=total_bits / len(raw),
    )
