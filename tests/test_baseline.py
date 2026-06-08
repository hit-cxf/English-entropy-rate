import tempfile
import unittest
from pathlib import Path

from english_entropy_rate.arithmetic import ArithmeticDecoder, ArithmeticEncoder
from english_entropy_rate.baseline import compression_baselines
from english_entropy_rate.my_llm_format import (
    HEADER_SIZE,
    MyLlmHeader,
    model_hash,
    pack_header,
    unpack_header,
)
from english_entropy_rate.normalize import normalize_to_lowercase_letters_and_spaces


class CompressionBaselineTests(unittest.TestCase):
    def test_compression_baselines_include_raw_and_standard_methods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("the quick brown fox jumps over the lazy dog\n" * 20)

            results = compression_baselines(path)
            methods = {result.method for result in results}

            self.assertEqual(methods, {"raw-utf8", "gzip", "bz2", "lzma"})
            self.assertTrue(all(result.bits_per_byte > 0 for result in results))
            self.assertTrue(all(result.bits_per_char > 0 for result in results))


class NormalizeTests(unittest.TestCase):
    def test_normalize_keeps_only_lowercase_letters_and_single_spaces(self) -> None:
        text = "Hello,\nWORLD!  It’s 2026."

        normalized = normalize_to_lowercase_letters_and_spaces(text)

        self.assertEqual(normalized, "hello world it s")
        self.assertRegex(normalized, r"^[a-z ]+$")
        self.assertNotIn("  ", normalized)


class ArithmeticCodingTests(unittest.TestCase):
    def test_arithmetic_round_trip_with_changing_distributions(self) -> None:
        symbols = [0, 2, 1, 1, 3, 0, 2]
        distributions = [
            [0, 3, 4, 7, 10],
            [0, 1, 5, 6, 10],
            [0, 2, 4, 9, 10],
            [0, 1, 2, 3, 10],
            [0, 4, 6, 8, 10],
            [0, 5, 6, 7, 10],
            [0, 1, 2, 8, 10],
        ]

        encoder = ArithmeticEncoder()
        for cumulative, symbol in zip(distributions, symbols):
            encoder.write(cumulative, symbol)
        payload = encoder.finish()

        decoder = ArithmeticDecoder(payload)
        decoded = [decoder.read(cumulative) for cumulative in distributions]

        self.assertEqual(decoded, symbols)


class MyLlmFormatTests(unittest.TestCase):
    def test_header_round_trip_is_fixed_size(self) -> None:
        header = MyLlmHeader(
            model_hash=model_hash("distilbert/distilgpt2"),
            original_size=123,
            token_count=45,
            first_token_id=6,
        )

        payload = pack_header(header)
        decoded = unpack_header(payload)

        self.assertEqual(len(payload), HEADER_SIZE)
        self.assertEqual(decoded, header)


if __name__ == "__main__":
    unittest.main()
