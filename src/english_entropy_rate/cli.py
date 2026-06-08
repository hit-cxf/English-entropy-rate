from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .baseline import compression_baselines
from .llm_estimator import estimate_llm_bits
from .normalize import normalize_file


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def run_baseline(args: argparse.Namespace) -> int:
    results = compression_baselines(args.path)
    rows = [
        [
            item.method,
            str(item.compressed_bytes),
            f"{item.compression_ratio:.4f}",
            f"{item.bits_per_byte:.4f}",
            f"{item.bits_per_char:.4f}",
        ]
        for item in results
    ]
    _print_table(
        ["method", "bytes", "ratio", "bits/byte", "bits/char"],
        rows,
    )
    return 0


def run_llm(args: argparse.Namespace) -> int:
    try:
        result = estimate_llm_bits(
            args.path,
            args.model,
            limit_chars=args.limit_chars,
            max_length=args.max_length,
            stride=args.stride,
            device=args.device,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows = [
        ["model", result.model],
        ["tokens_scored", str(result.tokens_scored)],
        ["chars", str(result.chars)],
        ["bytes", str(result.bytes)],
        ["total_bits", f"{result.total_bits:.2f}"],
        ["bits/token", f"{result.bits_per_token:.4f}"],
        ["bits/char", f"{result.bits_per_char:.4f}"],
        ["bits/byte", f"{result.bits_per_byte:.4f}"],
    ]
    _print_table(["metric", "value"], rows)
    return 0


def run_clean(args: argparse.Namespace) -> int:
    input_chars, output_chars = normalize_file(args.input, args.output)
    rows = [
        ["input", str(args.input)],
        ["output", str(args.output)],
        ["input_chars", str(input_chars)],
        ["output_chars", str(output_chars)],
    ]
    _print_table(["metric", "value"], rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="english-entropy-rate",
        description="Estimate English text entropy rate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser(
        "baseline",
        help="Measure raw/gzip/bz2/lzma compression rates.",
    )
    baseline.add_argument("path", type=Path)
    baseline.set_defaults(func=run_baseline)

    llm = subparsers.add_parser(
        "llm",
        help="Estimate ideal arithmetic-coding length from an LLM.",
    )
    llm.add_argument("path", type=Path)
    llm.add_argument("--model", default="distilgpt2")
    llm.add_argument(
        "--limit-chars",
        type=int,
        default=None,
        help="Score only the first N characters of the input file.",
    )
    llm.add_argument("--max-length", type=int, default=None)
    llm.add_argument("--stride", type=int, default=512)
    llm.add_argument("--device", default=None)
    llm.set_defaults(func=run_llm)

    clean = subparsers.add_parser(
        "clean",
        help="Create a text file containing only lowercase a-z and spaces.",
    )
    clean.add_argument("input", type=Path)
    clean.add_argument("output", type=Path)
    clean.set_defaults(func=run_clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
