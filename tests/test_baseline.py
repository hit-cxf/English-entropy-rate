import tempfile
import unittest
from pathlib import Path

from english_entropy_rate.baseline import compression_baselines
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


if __name__ == "__main__":
    unittest.main()
