from __future__ import annotations

from pathlib import Path


def normalize_to_lowercase_letters_and_spaces(text: str) -> str:
    """Keep only lowercase English letters and single spaces."""
    parts: list[str] = []
    previous_was_space = True

    for char in text.lower():
        if "a" <= char <= "z":
            parts.append(char)
            previous_was_space = False
        elif not previous_was_space:
            parts.append(" ")
            previous_was_space = True

    if parts and parts[-1] == " ":
        parts.pop()

    return "".join(parts)


def normalize_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    text = input_path.read_text(encoding="utf-8")
    normalized = normalize_to_lowercase_letters_and_spaces(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(normalized, encoding="ascii")
    return len(text), len(normalized)
