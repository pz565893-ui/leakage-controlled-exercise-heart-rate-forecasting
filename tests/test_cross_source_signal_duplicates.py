from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audit_cross_source_signal_duplicates import (  # noqa: E402
    DecodedSeries,
    exact_digest_from_decoded,
    lsh_signatures,
    make_lsh_projections,
    profile_variants,
    summarize_hr_profile,
    verify_near_candidate,
)


def make_series(
    session_key: str,
    hr: np.ndarray,
    *,
    hr_mask: np.ndarray | None = None,
    speed: np.ndarray | None = None,
    altitude: np.ndarray | None = None,
) -> DecodedSeries:
    n_bins = int(hr.size)
    if hr_mask is None:
        hr_mask = np.ones(n_bins, dtype=np.uint8)
    if speed is None:
        speed = np.linspace(8.0, 20.0, n_bins, dtype=np.float32)
    if altitude is None:
        altitude = np.linspace(50.0, 250.0, n_bins, dtype=np.float32)
    return DecodedSeries(
        session_key=session_key,
        grid_seconds=10,
        n_bins=n_bins,
        hr_values=hr.astype(np.float32),
        hr_mask=hr_mask.astype(np.uint8),
        altitude_values=altitude.astype(np.float32),
        altitude_mask=np.ones(n_bins, dtype=np.uint8),
        speed_values=speed.astype(np.float32),
        speed_mask=np.ones(n_bins, dtype=np.uint8),
    )


class CrossSourceSignalDuplicateTests(unittest.TestCase):
    def test_exact_digest_is_identifier_and_clock_translation_invariant(self) -> None:
        hr = np.linspace(90.0, 170.0, 240, dtype=np.float32)
        left = make_series("endomondo-1", hr)
        right = make_series("golden-user/session.csv", hr.copy())
        self.assertEqual(exact_digest_from_decoded(left), exact_digest_from_decoded(right))

    def test_exact_digest_changes_when_signal_changes(self) -> None:
        hr = np.linspace(90.0, 170.0, 240, dtype=np.float32)
        left = make_series("left", hr)
        changed = hr.copy()
        changed[120] += 1.0
        right = make_series("right", changed)
        self.assertNotEqual(exact_digest_from_decoded(left), exact_digest_from_decoded(right))

    def test_low_information_profiles_are_rejected(self) -> None:
        constant = np.full(240, 120.0, dtype=np.float32)
        summary = summarize_hr_profile(constant, np.ones(240, dtype=np.uint8))
        self.assertFalse(summary.eligible)
        self.assertEqual(summary.reason, "hr_variability_below_minimum")

    def test_lsh_is_deterministic_and_retains_small_perturbation_collision(self) -> None:
        time = np.linspace(0.0, 6.0 * math.pi, 48)
        profile = 130.0 + 20.0 * np.sin(time)
        perturbed = profile + 0.1 * np.sin(time * 4.0)
        projections = make_lsh_projections()
        left = lsh_signatures(profile, projections)
        right = lsh_signatures(perturbed, projections)
        self.assertEqual(left, lsh_signatures(profile.copy(), projections))
        self.assertGreaterEqual(sum(a == b for a, b in zip(left, right)), 1)

    def test_profile_and_continuous_verification_accept_small_perturbation(self) -> None:
        time = np.linspace(0.0, 8.0 * math.pi, 360)
        hr = (130.0 + 18.0 * np.sin(time) + 4.0 * np.sin(time / 3.0)).astype(
            np.float32
        )
        perturbed = (hr + 0.25 * np.sin(time * 2.0)).astype(np.float32)
        left = make_series("left", hr)
        right = make_series("right", perturbed)
        left_summary = summarize_hr_profile(left.hr_values, left.hr_mask)
        right_summary = summarize_hr_profile(right.hr_values, right.hr_mask)
        self.assertTrue(left_summary.eligible)
        self.assertTrue(right_summary.eligible)
        left_variants = profile_variants(left.hr_values, left.hr_mask, left_summary)
        right_variants = profile_variants(right.hr_values, right.hr_mask, right_summary)
        left_by_digest = {variant.digest: variant for variant in left_variants}
        shared = [variant for variant in right_variants if variant.digest in left_by_digest]
        self.assertTrue(shared)
        right_variant = shared[0]
        left_variant = left_by_digest[right_variant.digest]
        result = verify_near_candidate(
            left,
            right,
            [
                (
                    left_variant.crop_left_bins,
                    left_variant.crop_right_bins,
                    left_variant.quantization_offset_bpm,
                    right_variant.crop_left_bins,
                    right_variant.crop_right_bins,
                    right_variant.quantization_offset_bpm,
                )
            ],
        )
        self.assertEqual(result["verification_status"], "near_candidate_hr_plus_auxiliary")
        self.assertLess(float(result["hr_mae_bpm"]), 1.0)


if __name__ == "__main__":
    unittest.main()
