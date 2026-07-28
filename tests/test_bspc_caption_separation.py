from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_bspc_docx import remove_separate_caption_section  # noqa: E402


class BSPCCaptionSeparationTests(unittest.TestCase):
    def test_main_submission_omits_separately_supplied_captions(self) -> None:
        lines = [
            "## Acknowledgements",
            "Thanks.",
            "## Figure captions",
            "**Fig. 1. Design.** Caption text.",
            "**Supplementary Fig. 1. Audit.** Caption text.",
            "## Declaration of generative AI",
            "Disclosure.",
            "## References",
        ]
        filtered = remove_separate_caption_section(lines)
        self.assertNotIn("## Figure captions", filtered)
        self.assertFalse(any(line.startswith("**Fig.") for line in filtered))
        self.assertFalse(
            any(line.startswith("**Supplementary Fig.") for line in filtered)
        )
        self.assertIn("## Declaration of generative AI", filtered)
        self.assertIn("## References", filtered)


if __name__ == "__main__":
    unittest.main()
