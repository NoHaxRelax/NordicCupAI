"""End-to-end checks against saved public surveys and the Python reference.

Run after cargo build --release. Optional SEED_SEARCH_BINARY overrides the binary.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from search_survey_rust import DEFAULT_BINARY

BINARY = Path(os.environ.get("SEED_SEARCH_BINARY", DEFAULT_BINARY))


class IntegrationTests(unittest.TestCase):
    def scan(self, path, *arguments):
        return subprocess.run([str(BINARY), "--input", str(path), *map(str, arguments)],
                              capture_output=True, text=True)

    def test_final_surveys_match_python(self):
        for name in ("seed_survey_probe_2026-09-19.json", "seed_survey_heldout_2026-09-19.json"):
            path = ROOT / "docs" / name
            document = json.loads(path.read_text(encoding="utf-8"))
            for row in document["targets"]:
                with self.subTest(seed=row["target_seed"]):
                    process = self.scan(path, "--target", row["target_seed"], "--count", 100000, "--threads", 3)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    result = json.loads(process.stdout)
                    self.assertEqual(result["survivors"], row["checkpoints"][-1]["prefix_survivors"])
                    self.assertFalse(result["survivors_truncated"])
                    self.assertEqual(result["processed_seeds"], 100000)

    def test_earlier_checkpoint_matches_python(self):
        path = ROOT / "docs/seed_survey_heldout_2026-09-19.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        expected = next(frame["prefix_survivors"] for frame in document["targets"][0]["checkpoints"]
                        if frame["sim_time"] == 10.)
        process = self.scan(path, "--checkpoint", 10, "--count", 100000, "--threads", 3)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["survivors"], expected)

    def test_raw_samples_cannot_use_hidden_audit_or_record_seed(self):
        original = json.loads((ROOT / "docs/seed_survey_heldout_2026-09-19.json").read_text(encoding="utf-8"))
        row = original["targets"][0]
        # Only public samples are supplied; no target_seed, pose audit or action log.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text(json.dumps({"samples": row["checkpoints"][-1]["shared_samples"]}), encoding="utf-8")
            process = self.scan(path, "--count", 100000, "--threads", 3)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["survivors"], [78431])
            self.assertIsNone(result["selected_target_record"])

    def test_cli_ranges_truncation_and_output_protection(self):
        fixtures = json.loads(Path(__file__).with_name("python_vectors.json").read_text(encoding="utf-8"))
        fixture = fixtures["search_vectors"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            process = self.scan(path, "--count", fixture["count"], "--max-results", 2, "--threads", 3)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertTrue(result["complete"])
            self.assertTrue(result["survivors_truncated"])
            self.assertEqual(result["survivor_count"], len(fixture["expected"]))
            self.assertEqual(result["survivors"], fixture["expected"][:2])
            for arguments in [("--count", 0), ("--start", 2**32 - 1, "--count", 2),
                              ("--threads", 0), ("--checkpoint", "NaN"), ("--output", path)]:
                process = self.scan(path, *arguments)
                self.assertNotEqual(process.returncode, 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), fixture)

    def test_runner_does_not_certify_truncated_or_unverified_candidates(self):
        for extra, reason in [((), "Native-world cap"), (("--max-results", "2"), "truncated")]:
            process = subprocess.run([
                sys.executable, str(ROOT / "scripts/search_survey_rust.py"),
                "--binary", str(BINARY),
                "--input", str(ROOT / "docs/seed_survey_heldout_2026-09-19.json"),
                "--checkpoint", "10", "--count", "100000", "--threads", "3", "--verify", *extra,
            ], capture_output=True, text=True)
            self.assertEqual(process.returncode, 2, process.stderr)
            result = json.loads(process.stdout)
            self.assertTrue(result["complete"])
            self.assertEqual(result["survivor_count"], 173)
            self.assertFalse(result["verification_complete"])
            self.assertIsNone(result["unique_verified_seed_in_range"])
            self.assertEqual(result["verified_seeds"], [])
            self.assertIn(reason, result["verification_skipped"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
