from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass


MAGIC = b"MYLLM001"
HEADER_STRUCT = struct.Struct(">8s32sQII")
HEADER_SIZE = HEADER_STRUCT.size


@dataclass(frozen=True)
class MyLlmHeader:
    model_hash: bytes
    original_size: int
    token_count: int
    first_token_id: int


def model_hash(model_name: str) -> bytes:
    return hashlib.sha256(model_name.encode("utf-8")).digest()


def pack_header(header: MyLlmHeader) -> bytes:
    if len(header.model_hash) != 32:
        raise ValueError("model_hash must be 32 bytes")
    return HEADER_STRUCT.pack(
        MAGIC,
        header.model_hash,
        header.original_size,
        header.token_count,
        header.first_token_id,
    )


def unpack_header(payload: bytes) -> MyLlmHeader:
    if len(payload) < HEADER_SIZE:
        raise ValueError("payload is too short to contain a .my-llm header")
    magic, hash_bytes, original_size, token_count, first_token_id = HEADER_STRUCT.unpack(
        payload[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError("invalid .my-llm magic header")
    return MyLlmHeader(
        model_hash=hash_bytes,
        original_size=original_size,
        token_count=token_count,
        first_token_id=first_token_id,
    )
