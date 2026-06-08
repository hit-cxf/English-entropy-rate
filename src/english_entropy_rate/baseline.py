from __future__ import annotations

import bz2
import gzip
import lzma
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompressionResult:
    method: str
    compressed_bytes: int
    bits_per_byte: float
    bits_per_char: float
    compression_ratio: float


def read_text_bytes(path: Path) -> tuple[str, bytes]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    return text, raw


def compression_baselines(path: Path) -> list[CompressionResult]:
    text, raw = read_text_bytes(path)
    byte_count = len(raw)
    char_count = len(text)
    if byte_count == 0 or char_count == 0:
        raise ValueError("input text must not be empty")

    compressed = {
        "raw-utf8": raw,
        "gzip": gzip.compress(raw, compresslevel=9),
        "bz2": bz2.compress(raw, compresslevel=9),
        "lzma": lzma.compress(raw, preset=9),
    }

    return [
        CompressionResult(
            method=method,
            compressed_bytes=len(payload),
            bits_per_byte=8.0 * len(payload) / byte_count,
            bits_per_char=8.0 * len(payload) / char_count,
            compression_ratio=len(payload) / byte_count,
        )
        for method, payload in compressed.items()
    ]
