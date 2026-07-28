from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_pmea_docx import normalize_submission_typography, replace_citations  # noqa: E402
from build_pmea_supplementary_docx import filter_for_pmea  # noqa: E402


class PmeaSubmissionAdaptationTests(unittest.TestCase):
    def test_email_addresses_are_not_parsed_as_bibtex_citations(self) -> None:
        source = "Corresponding author: 20248657@o.shinhan.ac.kr"

        converted, cited, labels = replace_citations(source, {})

        self.assertEqual(converted, source)
        self.assertEqual(cited, [])
        self.assertEqual(labels, {})

    def test_typography_converts_ranges_and_negative_signs_without_breaking_tables(self) -> None:
        source = (
            "0.029--0.112 bpm; 2.7%--13.9%; 95% CI -0.056 to 0.006\n"
            "| --- | ---: |"
        )

        converted = normalize_submission_typography(source)

        self.assertEqual(
            converted,
            "0.029–0.112 bpm; 2.7%–13.9%; 95% CI −0.056 to 0.006\n"
            "| --- | ---: |",
        )

    def test_supplement_filter_keeps_only_reportable_joint_sport_rows(self) -> None:
        source = [
            "### Table S5a. Point forecasts",
            "| Regime | Held sport | 1 min | 3 min | 5 min | Seeds | Users | Sessions | Origins | Support |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| Joint user–sport | outdoor cycling | 8.1 | 9.9 | 10.9 | 3 | 77 | 6097 | 37856 | Supported |",
            "| Joint user–sport | walking hiking | 6.5 | 7.5 | 8.2 | 3 | 19 | 96 | 428 | Caution |",
            "| Same-user unseen sport | walking hiking | 6.0 | 7.0 | 8.0 | 3 | 120 | 900 | 5000 | Supported |",
        ]

        filtered = filter_for_pmea(source)
        joined = "\n".join(filtered)

        self.assertIn("| Joint user–sport | outdoor cycling", joined)
        self.assertNotIn("| Joint user–sport | walking hiking", joined)
        self.assertIn("| Same-user unseen sport | walking hiking", joined)

    def test_supplement_filter_removes_recorded_gender_outcomes(self) -> None:
        source = [
            "## Table S8. Recorded-gender outcomes",
            "gender outcome row",
            "## Table S9. Next analysis",
            "retained row",
        ]

        filtered = filter_for_pmea(source)

        self.assertEqual(filtered, ["## Table S8. Next analysis", "retained row"])

    def test_supplement_filter_renumbers_all_later_table_references(self) -> None:
        source = [
            "## Table S8. Recorded-gender outcomes",
            "gender outcome row",
            "## Table S9. First retained analysis",
            "See Supplementary Tables S11 and S19b.",
        ]

        filtered = filter_for_pmea(source)

        self.assertEqual(filtered[0], "## Table S8. First retained analysis")
        self.assertEqual(filtered[1], "See Supplementary Tables S10 and S18b.")

    def test_supplement_filter_retargets_title_and_journal(self) -> None:
        source = [
            "## Uncertainty-Aware Exercise Heart-Rate Forecasting under User and Sport Distribution Shifts: A Leakage-Controlled Multi-Dataset Study",
            "**Target journal:** *Biomedical Signal Processing and Control*",
        ]

        joined = "\n".join(filter_for_pmea(source))

        self.assertIn("Boundary-dependent reliability", joined)
        self.assertIn("*Physiological Measurement*", joined)
        self.assertNotIn("Biomedical Signal Processing and Control", joined)


if __name__ == "__main__":
    unittest.main()
