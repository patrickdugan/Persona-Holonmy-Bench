"""Deterministic offline tests for the Resolution probe-family audit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resolution_probe_transfer.py"
spec = importlib.util.spec_from_file_location("resolution_probe_transfer", SCRIPT)
assert spec is not None and spec.loader is not None
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class ProbeTransferTests(unittest.TestCase):
    def fixture(self, root: Path):
        r = {"flaky_ci": 0.1, "parser_validator": 0.2, "stale_cache": -0.1}
        l = {"flaky_ci": 0.446, "parser_validator": 0.616, "stale_cache": -0.020}
        for genre in transfer.GENRES:
            path = root / "reports" / "cal9200" / genre / "summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"groups": {f"{genre}/default": {
                "n_seeds": 16,
                "contrasts": {
                    "exit/ordered-neutral": {"copy_free_margin": {"per_probe_mean": r}},
                    "exit/lexical_match-lexical_neutral": {"copy_free_margin": {"per_probe_mean": l}},
                },
            }}}), encoding="utf-8")
        path = root / "reports" / "confirmatory.md"
        path.write_text("""**4b_confirmatory**
| probe family | R | L | F | P |
|---|---|---|---|---|
| feature_flag | +0.296 | +0.145 | +0.754 | +0.151 |
| log_rotation | +0.331 | +0.074 | -0.002 | +0.257 |
| memory_leak | +0.679 | -0.096 | +0.759 | +0.774 |
| rate_limit | -0.280 | +0.117 | +0.279 | -0.396 |
| retry_backoff | +0.584 | +0.154 | +0.862 | +0.430 |
| schema_migration | +0.664 | +0.304 | +0.267 | +0.360 |

- 17b_confirmatory: missing
""", encoding="utf-8")

    def test_exact_signflip(self):
        self.assertEqual(transfer.exact_signflip_p([-1, -1, -1]), 0.25)
        with self.assertRaises(ValueError):
            transfer.exact_signflip_p([])

    def test_audit_sign_transfer_and_determinism(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            self.fixture(root)
            first = transfer.audit(root)
            second = transfer.audit(root)
            self.assertEqual(first, second)
            self.assertEqual(first["calibration_9200"]["n_negative"], 3)
            self.assertAlmostEqual(first["calibration_9200"]["mean"], -0.2806666667)
            self.assertEqual(first["confirmatory_4b"]["n_positive"], 5)
            self.assertEqual(first["cross_stratum"]["confirmatory_families_matching_calibration_mean_sign"], 1)
            self.assertAlmostEqual(first["confirmatory_4b"]["mean"], 0.2626666667)
            self.assertAlmostEqual(first["confirmatory_4b"]["exact_two_sided_family_signflip_p"], 0.15625)

    def test_missing_probe_fails_closed(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            self.fixture(root)
            report_path = root / "reports" / "confirmatory.md"
            report_path.write_text(report_path.read_text(encoding="utf-8").replace(
                "| rate_limit | -0.280 | +0.117 | +0.279 | -0.396 |\n", ""
            ), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing confirmatory probes"):
                transfer.audit(root)

    def test_inconsistent_contrast_fails_closed(self):
        with TemporaryDirectory() as work:
            root = Path(work)
            self.fixture(root)
            report_path = root / "reports" / "confirmatory.md"
            report_path.write_text(report_path.read_text(encoding="utf-8").replace(
                "+0.151 |", "+0.900 |"
            ), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Reported P"):
                transfer.audit(root)


if __name__ == "__main__":
    unittest.main()
