import tempfile
import unittest
from pathlib import Path

from english_entropy_rate.baseline import compression_baselines


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


if __name__ == "__main__":
    unittest.main()
