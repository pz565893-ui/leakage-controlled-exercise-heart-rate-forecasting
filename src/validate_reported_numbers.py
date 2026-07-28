"""Validate manuscript and supplement numbers against frozen authoritative artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
Q1 = ROOT / "outputs" / "q1_multiseed_v0_21_0" / "aggregation"
ZERO = ROOT / "outputs" / "independent_zero_history_v0_23_0" / "aggregation"
LEAKY = ROOT / "outputs" / "deliberately_leaky_negative_control_v0_28_0" / "aggregation"
MAIN = ROOT / "manuscript" / "main_manuscript.md"
SUPPLEMENT = ROOT / "manuscript" / "supplementary_material.md"
REPORT = ROOT / "outputs" / "audit" / "REPORTED_NUMBER_VALIDATION.json"
HORIZONS = [60, 180, 300]


def read_result(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def table_one_rows(text: str) -> dict[str, list[str]]:
    match = re.search(
        r"\| Regime and model \| 1 min \| 3 min \| 5 min \|\n"
        r"\|[-: |]+\|\n(?P<body>(?:\|.*\|\n)+)",
        text,
    )
    if not match:
        return {}
    rows: dict[str, list[str]] = {}
    for line in match.group("body").splitlines():
        cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
        if len(cells) == 4:
            rows[cells[0]] = cells[1:]
    return rows


def q1_metric(
    frame: pd.DataFrame,
    *,
    experiment: str,
    regime: str,
    mode: str,
    metric: str,
    source_kind: str,
    nominal: float | None = None,
    calibrated: bool | None = None,
) -> pd.DataFrame:
    selected = frame[
        (frame.experiment == experiment)
        & (frame.regime == regime)
        & (frame["mode"] == mode)
        & (frame.metric == metric)
        & (frame.source_kind == source_kind)
    ].copy()
    if nominal is not None:
        selected = selected[selected.nominal_coverage == nominal]
    if calibrated is not None:
        selected = selected[selected.calibrated == calibrated]
    selected = selected.set_index("horizon_seconds").reindex(HORIZONS).reset_index()
    if selected.value_median.isna().any():
        raise AssertionError(
            f"Incomplete Q1 selection: {experiment}/{regime}/{mode}/{metric}"
        )
    return selected


def independent_point(per_seed: pd.DataFrame, protocol: str, evaluation: str) -> pd.DataFrame:
    selected = per_seed[
        (per_seed.protocol == protocol) & (per_seed.evaluation == evaluation)
    ][["seed", "horizon_seconds", "independent_zero_mae_bpm"]].drop_duplicates()
    summary = (
        selected.groupby("horizon_seconds").independent_zero_mae_bpm
        .agg(["median", "min", "max"])
        .reindex(HORIZONS)
        .reset_index()
    )
    if summary["median"].isna().any():
        raise AssertionError(f"Incomplete independent selection: {protocol}/{evaluation}")
    return summary


def table_range_rows(frame: pd.DataFrame) -> list[str]:
    return [
        f"{row.value_median:.3f} [{row.value_minimum:.3f}--{row.value_maximum:.3f}]"
        for _, row in frame.iterrows()
    ]


def independent_table_rows(frame: pd.DataFrame) -> list[str]:
    return [
        f"{row['median']:.3f} [{row['min']:.3f}--{row['max']:.3f}]"
        for _, row in frame.iterrows()
    ]


def fixed_rows(frame: pd.DataFrame, *, model: str) -> list[str]:
    selected = frame[frame.model == model].set_index("horizon_seconds").reindex(HORIZONS)
    return [f"{value:.3f}" for value in selected.mae_bpm]


def sup_range(row: pd.Series, digits: int) -> str:
    return (
        f"{float(row.value_median):.{digits}f} "
        f"[{float(row.value_minimum):.{digits}f} to "
        f"{float(row.value_maximum):.{digits}f}]"
    )


def main() -> int:
    main_text = MAIN.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    q1 = pd.read_csv(Q1 / "seed_variability_summary_v0_22_0.csv")
    zero_per_seed = pd.read_csv(ZERO / "strategy_contrasts_per_seed_v0_23_0.csv")
    zero_bootstrap = pd.read_csv(
        ZERO / "strategy_contrast_user_bootstrap_v0_23_0.csv"
    )
    external_standardization = read_result("external_sport_standardization_v0_20_1.csv")
    external_sport_uncertainty = read_result(
        "external_sport_uncertainty_bootstrap_v0_21_0.csv"
    )
    balanced_calibration = read_result(
        "multiseed_balanced_calibration_summary_v0_24_0.csv"
    )
    uncertainty_standardization = read_result(
        "external_sport_uncertainty_standardization_v0_24_0.csv"
    )
    paired_models_v025 = read_result(
        "multiseed_paired_model_comparisons_v0_25_0.csv"
    )
    paired_sports_v025 = read_result(
        "multiseed_paired_sport_shift_v0_25_0.csv"
    )
    source_characterization = read_result("source_shift_characterization_v0_21_0.csv")
    persistence_conformal = read_result("persistence_conformal_baseline_v0_26_0.csv")
    matched_sport = read_result("matched_sport_availability_v0_27_0.csv")
    leaky_seed_summary = pd.read_csv(LEAKY / "paired_metrics_seed_summary_v0_28_0.csv")
    leaky_bootstrap = pd.read_csv(LEAKY / "paired_user_bootstrap_v0_28_0.csv")
    leaky_intervals = pd.read_csv(LEAKY / "interval_diagnostics_per_seed_v0_28_0.csv")
    horizon_eligibility = read_result("horizon_specific_eligibility_v0_29_0.csv")
    horizon_frozen = read_result(
        "horizon_specific_frozen_model_summary_v0_30_0.csv"
    )
    errors: list[str] = []
    checks: dict[str, bool] = {}

    def record(key: str, ok: bool, message: str) -> None:
        checks[key] = bool(ok)
        if not ok:
            errors.append(message)

    temporal_base = read_result("temporal_aligned_baselines_v0_13_0.csv")
    naive = read_result("naive_baseline_metrics_v0_5_0.csv")
    expected_table = {
        "Strict temporal, history-informed": table_range_rows(
            q1_metric(
                q1,
                experiment="temporal_main",
                regime="within_user_temporal_test",
                mode="history_informed",
                metric="mae_bpm",
                source_kind="point",
            )
        ),
        "Strict temporal, zero-history-trained": independent_table_rows(
            independent_point(zero_per_seed, "strict_temporal", "internal_test")
        ),
        "Strict temporal, GRU": table_range_rows(
            q1_metric(
                q1,
                experiment="temporal_gru",
                regime="within_user_temporal_test",
                mode="not_applicable",
                metric="mae_bpm",
                source_kind="point",
            )
        ),
        "Strict temporal, persistence": fixed_rows(temporal_base, model="persistence"),
        "Strict temporal, EWMA": fixed_rows(temporal_base, model="ewma_alpha_0_1"),
        "Unseen user, history-informed": table_range_rows(
            q1_metric(
                q1,
                experiment="unseen_main",
                regime="unseen_user_test",
                mode="history_informed",
                metric="mae_bpm",
                source_kind="point",
            )
        ),
        "Unseen user, zero-history-trained": independent_table_rows(
            independent_point(zero_per_seed, "unseen_user", "internal_test")
        ),
        "Unseen user, GRU": table_range_rows(
            q1_metric(
                q1,
                experiment="unseen_gru",
                regime="unseen_user_test",
                mode="not_applicable",
                metric="mae_bpm",
                source_kind="point",
            )
        ),
        "Unseen user, persistence": fixed_rows(
            naive[naive.regime == "unseen_user_test"], model="persistence"
        ),
        "Unseen user, EWMA": fixed_rows(
            naive[naive.regime == "unseen_user_test"], model="ewma"
        ),
        "GoldenCheetah, history-masked": table_range_rows(
            q1_metric(
                q1,
                experiment="unseen_main",
                regime="goldencheetah_frozen_external",
                mode="zero_history",
                metric="mae_bpm",
                source_kind="point",
            )
        ),
        "GoldenCheetah, zero-history-trained": independent_table_rows(
            independent_point(zero_per_seed, "unseen_user", "frozen_external")
        ),
        "GoldenCheetah, GRU": table_range_rows(
            q1_metric(
                q1,
                experiment="unseen_gru",
                regime="goldencheetah_frozen_external",
                mode="not_applicable",
                metric="mae_bpm",
                source_kind="point",
            )
        ),
        "GoldenCheetah, persistence": fixed_rows(
            naive[naive.regime == "goldencheetah_frozen_external"],
            model="persistence",
        ),
        "GoldenCheetah, EWMA": fixed_rows(
            naive[naive.regime == "goldencheetah_frozen_external"], model="ewma"
        ),
    }
    reported_table = table_one_rows(main_text)
    for label, expected in expected_table.items():
        actual = reported_table.get(label)
        record(
            f"table_1::{label}",
            actual == expected,
            f"Table 1 mismatch for {label}: reported={actual}, expected={expected}",
        )

    required_main = [
        "7,635,176 (37.4%)",
        "101,184 origins from 15,026 sessions and 105 users",
        "104,144 origins from 16,012 sessions and 948 users",
        "531,725 origins from 31,851 sessions and 144 users",
        "14,925 of 15,026 unseen-user test sessions (99.3%",
        "Relative to separately trained zero-history models",
        "-0.029 bpm (95% CI -0.056 to 0.006)",
        "-0.006 bpm (95% CI -0.015 to 0.003)",
        "0.896, 0.897, and 0.892",
        "0.892, 0.887, and 0.877",
        "0.880, 0.859, and 0.850",
        "conditional on frozen reference seed 20260722",
        "does not furnish a finite-sample guarantee for the session-then-user PICP estimand",
        "we make no claim of global priority",
        "506,203 predictions, 5.00 times the 101,184 standard unseen-user origins",
        "same complete-three-target cohort",
        "therefore did not evaluate cross-source transport of the completed-history branch",
        "AdamW (initial learning rate 0.001; weight decay 0.0001)",
        "The final analysis evaluated three five-seed strategy contrasts",
        "Versioned configurations and per-checkpoint freezing improve traceability but do not replace preregistration",
        "Reference-seed GoldenCheetah results were inspected before the final five-seed reporting plan was consolidated",
    ]
    for phrase in required_main:
        record(
            f"main_phrase::{phrase}",
            phrase in main_text,
            f"Required manuscript statement missing: {phrase}",
        )

    equal_family = external_standardization[
        external_standardization.comparison_scope == "equal_family_standardized"
    ].set_index("horizon_seconds").reindex(HORIZONS)
    expected_standardization = [
        (
            f"{float(row.external_minus_internal_bpm):.3f}"
            + (" bpm" if int(horizon) == 60 else "")
            + (
                f" (95% CI {float(row.difference_ci_low_bpm):.3f} to "
                f"{float(row.difference_ci_high_bpm):.3f})"
                if int(horizon) == 60
                else (
                    f" ({float(row.difference_ci_low_bpm):.3f} to "
                    f"{float(row.difference_ci_high_bpm):.3f})"
                )
            )
        )
        for horizon, row in equal_family.iterrows()
    ]
    for horizon, token in zip(HORIZONS, expected_standardization):
        record(
            f"main_external_standardization::{horizon}",
            token in main_text,
            f"Missing main-text equal-family standardization token: {token}",
        )

    sport_5min = external_sport_uncertainty[
        external_sport_uncertainty.horizon_seconds == 300
    ].set_index("sport_family")
    for sport in ["indoor_virtual_cycling", "outdoor_cycling", "running"]:
        token = f"{float(sport_5min.loc[sport, 'picp_90']):.3f}"
        record(
            f"supplement_external_sport_picp_5min::{sport}",
            token in supplement,
            f"Missing supplementary external sport PICP token for {sport}: {token}",
        )

    calibration_90 = balanced_calibration[
        (balanced_calibration.regime == "goldencheetah_frozen_external")
        & (balanced_calibration["mode"] == "zero_history")
        & (balanced_calibration.nominal_coverage == 0.9)
        & (
            balanced_calibration.calibration_method
            == "equal_user_equal_session_empirical"
        )
    ].set_index("horizon_seconds").reindex(HORIZONS)
    calibration_token = (
        "increased external 90% PICP to "
        + ", ".join(f"{float(value):.3f}" for value in calibration_90.picp_median.iloc[:2])
        + f", and {float(calibration_90.picp_median.iloc[2]):.3f}"
    )
    record(
        "main_multiseed_balanced_external_picp",
        calibration_token in main_text,
        f"Missing main-text multiseed calibration sensitivity token: {calibration_token}",
    )

    wis = uncertainty_standardization[
        uncertainty_standardization.metric == "weighted_interval_score"
    ]
    for scope in ["shared_family_natural_mix", "equal_family_standardized"]:
        selected = wis[wis.comparison_scope == scope].set_index(
            "horizon_seconds"
        ).reindex(HORIZONS)
        token = ", ".join(
            f"{float(value):.3f}" for value in selected.external_minus_internal.iloc[:2]
        ) + f", and {float(selected.external_minus_internal.iloc[2]):.3f} bpm"
        record(
            f"main_interval_standardization_wis::{scope}",
            token in main_text,
            f"Missing main-text WIS standardization token for {scope}: {token}",
        )

    persistence_5min = persistence_conformal[
        persistence_conformal.horizon_seconds == 300
    ].set_index("regime").reindex(
        [
            "strict_temporal_test",
            "unseen_user_test",
            "goldencheetah_frozen_external",
        ]
    )
    persistence_main_tokens = {
        "wis": "WIS values of "
        + ", ".join(
            f"{float(value):.3f}"
            for value in persistence_5min.weighted_interval_score.iloc[:2]
        )
        + f", and {float(persistence_5min.weighted_interval_score.iloc[2]):.3f}",
        "picp": "Its 90% PICP was "
        + ", ".join(
            f"{float(value):.3f}" for value in persistence_5min.picp_90.iloc[:2]
        )
        + f", and {float(persistence_5min.picp_90.iloc[2]):.3f}",
    }
    for label, token in persistence_main_tokens.items():
        record(
            f"main_persistence_conformal::{label}",
            token in main_text,
            f"Missing main-text persistence-conformal token: {token}",
        )

    matched_5min = matched_sport[matched_sport.horizon_seconds == 300]
    sport_display = {
        "outdoor_cycling": "Outdoor cycling",
        "indoor_virtual_cycling": "Indoor/virtual cycling",
        "running": "Running",
        "walking_hiking": "Walking/hiking",
        "strength_cross_training": "Strength/cross-training",
    }
    reportable_position = 0
    for _, row in matched_5min.iterrows():
        users = int(row.users)
        effect = float(row.held_minus_full_delta_mae_bpm)
        ci_low = float(row.ci_low_bpm)
        ci_high = float(row.ci_high_bpm)
        sport = str(row.held_sport_family)

        source_token = (
            f"| {sport_display[sport]} | 5 min | "
            f"{float(row.full_sport_mae_bpm):.3f} / "
            f"{float(row.held_sport_mae_bpm):.3f} | "
            f"{effect:.3f} [{ci_low:.3f}, {ci_high:.3f}] |"
        )
        record(
            f"source_supplement_matched_sport::{sport}",
            source_token in supplement,
            f"Missing source-supplement matched-sport row: {source_token}",
        )

        if users >= 30:
            if reportable_position == 0:
                token = (
                    f"{effect:.3f} bpm (95% CI {ci_low:.3f} to "
                    f"{ci_high:.3f}; {users} users)"
                )
            else:
                token = f"{effect:.3f} ({ci_low:.3f} to {ci_high:.3f}; {users} users)"
            record(
                f"main_matched_sport_reportable::{sport}",
                token in main_text,
                f"Missing reportable main-text matched-sport token: {token}",
            )
            reportable_position += 1
        else:
            disallowed_tokens = (
                f"{effect:.3f} bpm (95% CI {ci_low:.3f} to {ci_high:.3f})",
                f"{effect:.3f} ({ci_low:.3f} to {ci_high:.3f})",
            )
            record(
                f"main_matched_sport_low_support_omitted::{sport}",
                not any(token in main_text for token in disallowed_tokens),
                "Low-support matched-sport outcome was found in journal-facing main text: "
                + " OR ".join(disallowed_tokens),
            )

    leaky_point_bootstrap = leaky_bootstrap[
        (leaky_bootstrap.metric_family == "point")
        & (leaky_bootstrap.metric == "mae_bpm")
    ].set_index("horizon_seconds").reindex(HORIZONS)
    for position, (horizon, row) in enumerate(leaky_point_bootstrap.iterrows()):
        if position == 0:
            token = (
                f"{float(row.leaky_minus_clean_estimate):.3f} bpm "
                f"(95% CI {float(row.ci_low):.3f} to {float(row.ci_high):.3f})"
            )
        else:
            token = (
                f"{float(row.leaky_minus_clean_estimate):.3f} "
                f"({float(row.ci_low):.3f} to {float(row.ci_high):.3f})"
            )
        record(
            f"main_leaky_negative_control::mae::{int(horizon)}",
            token in main_text,
            f"Missing main-text deliberately leaky MAE token: {token}",
        )
    for token in (
        "15,839 of 16,012 test sessions entered fitting",
        "98.9% of test origins",
        "95.0% shared at least one target timestamp",
    ):
        record(
            f"main_leaky_negative_control::design::{token}",
            token in main_text,
            f"Missing main-text deliberately leaky design token: {token}",
        )

    characterization = source_characterization.set_index("metric")
    source_tokens = {
        "duration": (
            f"{float(characterization.loc['session_duration_minutes', 'external_point']):.1f} "
            f"versus {float(characterization.loc['session_duration_minutes', 'internal_point']):.1f} min"
        ),
        "sampling_gap": (
            f"{float(characterization.loc['raw_median_sampling_gap_seconds', 'external_point']):.2f} "
            f"versus {float(characterization.loc['raw_median_sampling_gap_seconds', 'internal_point']):.2f} s"
        ),
    }
    for label, token in source_tokens.items():
        record(
            f"main_source_characterization::{label}",
            token in main_text,
            f"Missing main-text source characterization token: {token}",
        )

    supplement_reproducibility = [
        "the last valid HR, speed, or altitude value in each bin was retained",
        "The 13 completed-history variables were",
        "Formal main runs used AdamW",
        "the held family was absent from training, validation, calibration, and history updates",
        "The session-eligible split contained 1,090 Endomondo users",
    ]
    for phrase in supplement_reproducibility:
        record(
            f"supplement_reproducibility::{phrase}",
            phrase in supplement,
            f"Required supplementary reproducibility statement missing: {phrase}",
        )

    prohibited = [
        "an independently trained no-history model was not executed",
        "All final learned models also used one training seed",
        "Joint user--sport PICP ranged from 0.855 to 0.936",
        "12.273 bpm (95% CI 8.835--16.860)",
        "History-informed hierarchical MAE is reported with 95% user-bootstrap confidence intervals",
        "paired_user_bootstrap_v0_11_0.csv` is the authoritative",
    ]
    joined = main_text + "\n" + supplement
    for phrase in prohibited:
        record(
            f"stale_absent::{phrase}",
            phrase not in joined,
            f"Stale or prohibited statement present: {phrase}",
        )

    for _, row in zero_bootstrap.iterrows():
        token = (
            f"{float(row.estimate_bpm):.3f} "
            f"[{float(row.percentile_95_ci_low_bpm):.3f}, "
            f"{float(row.percentile_95_ci_high_bpm):.3f}]"
        )
        key = (
            f"zero_bootstrap::{row.protocol}::{row.evaluation}::"
            f"{row.contrast}::{int(row.horizon_seconds)}"
        )
        record(key, token in supplement, f"Missing v0.23 contrast token for {key}: {token}")

    for _, row in paired_models_v025.iterrows():
        token = (
            f"{float(row.delta_mae_bpm):.3f} "
            f"[{float(row.ci_low_bpm):.3f}, {float(row.ci_high_bpm):.3f}]"
        )
        key = (
            f"v025_model::{row.regime}::{row.comparator_model}::"
            f"{int(row.horizon_seconds)}"
        )
        record(
            key,
            token in supplement,
            f"Missing v0.25 paired model token for {key}: {token}",
        )

    for _, row in paired_sports_v025.iterrows():
        token = (
            f"{float(row.delta_mae_bpm):.3f} "
            f"[{float(row.ci_low_bpm):.3f}, {float(row.ci_high_bpm):.3f}]"
        )
        key = (
            f"v025_sport::{row.regime}::{row.held_sport_family}::"
            f"{int(row.horizon_seconds)}"
        )
        record(
            key,
            token in supplement,
            f"Missing v0.25 paired sport token for {key}: {token}",
        )

    for _, row in persistence_conformal.iterrows():
        key_root = f"v026_persistence::{row.regime}::{int(row.horizon_seconds)}"
        expected = {
            "mae": f"{float(row.mae_bpm):.3f}",
            "picp": " / ".join(
                f"{float(row[f'picp_{level}']):.3f}" for level in (50, 80, 90)
            ),
            "width": " / ".join(
                f"{float(row[f'width_{level}_bpm']):.2f}" for level in (50, 80, 90)
            ),
            "wis": f"{float(row.weighted_interval_score):.3f}",
        }
        for metric, token in expected.items():
            record(
                f"{key_root}::{metric}",
                token in supplement,
                f"Missing v0.26 persistence token for {key_root}/{metric}: {token}",
            )

    for _, row in matched_sport.iterrows():
        key_root = f"v027_matched_sport::{row.held_sport_family}::{int(row.horizon_seconds)}"
        expected = {
            "mae": (
                f"{float(row.full_sport_mae_bpm):.3f} / "
                f"{float(row.held_sport_mae_bpm):.3f}"
            ),
            "delta": (
                f"{float(row.held_minus_full_delta_mae_bpm):.3f} "
                f"[{float(row.ci_low_bpm):.3f}, {float(row.ci_high_bpm):.3f}]"
            ),
            "user_percent": f"{float(row.users_with_higher_held_error_percent):.1f}%",
            "support": (
                f"{int(row.users):,} / {int(row.sessions):,} / {int(row.origins):,}"
            ),
        }
        for metric, token in expected.items():
            record(
                f"{key_root}::{metric}",
                token in supplement,
                f"Missing v0.27 matched-sport token for {key_root}/{metric}: {token}",
            )

    leaky_summary_index = leaky_seed_summary.set_index("horizon_seconds")
    for horizon in HORIZONS:
        summary = leaky_summary_index.loc[horizon]
        effect = leaky_point_bootstrap.loc[horizon]
        expected = {
            "clean_mae": (
                f"{float(summary.clean_mae_bpm_median):.3f} "
                f"[{float(summary.clean_mae_bpm_minimum):.3f} to "
                f"{float(summary.clean_mae_bpm_maximum):.3f}]"
            ),
            "leaky_mae": (
                f"{float(summary.leaky_mae_bpm_median):.3f} "
                f"[{float(summary.leaky_mae_bpm_minimum):.3f} to "
                f"{float(summary.leaky_mae_bpm_maximum):.3f}]"
            ),
            "delta_mae": (
                f"{float(effect.leaky_minus_clean_estimate):.3f} "
                f"[{float(effect.ci_low):.3f}, {float(effect.ci_high):.3f}]"
            ),
            "relative_optimism": (
                f"{float(summary.relative_mae_optimism_percent_median):.3f} "
                f"[{float(summary.relative_mae_optimism_percent_minimum):.3f} to "
                f"{float(summary.relative_mae_optimism_percent_maximum):.3f}]"
            ),
        }
        for metric, token in expected.items():
            record(
                f"v028_leaky::{int(horizon)}::{metric}",
                token in supplement,
                f"Missing v0.28 deliberately leaky token for {horizon}/{metric}: {token}",
            )

    leaky_90 = leaky_intervals[
        (leaky_intervals.nominal_coverage == 0.9)
        & (leaky_intervals.calibrated == True)
    ]
    leaky_interval_bootstrap = leaky_bootstrap[
        (leaky_bootstrap.metric_family == "interval")
        & (leaky_bootstrap.nominal_coverage == 0.9)
        & (leaky_bootstrap.calibrated == True)
    ]
    for horizon in HORIZONS:
        seed_rows = leaky_90[leaky_90.horizon_seconds == horizon]
        for metric in ("picp", "mean_interval_width_bpm"):
            row = leaky_interval_bootstrap[
                (leaky_interval_bootstrap.horizon_seconds == horizon)
                & (leaky_interval_bootstrap.metric == metric)
            ].iloc[0]
            token = (
                f"{float(row.leaky_minus_clean_estimate):.3f} "
                f"[{float(row.ci_low):.3f}, {float(row.ci_high):.3f}]"
            )
            record(
                f"v028_leaky::{int(horizon)}::{metric}",
                token in supplement,
                f"Missing v0.28 interval token for {horizon}/{metric}: {token}",
            )
        record(
            f"v028_leaky::{int(horizon)}::picp_medians",
            (
                f"{float(seed_rows.clean_picp.median()):.3f} / "
                f"{float(seed_rows.leaky_picp.median()):.3f}"
            )
            in supplement,
            f"Missing v0.28 PICP medians for horizon {horizon}",
        )

    horizon_specific = horizon_eligibility[
        horizon_eligibility.cohort == "horizon_specific"
    ]
    for _, row in horizon_specific.iterrows():
        common = horizon_eligibility[
            (horizon_eligibility.regime == row.regime)
            & (horizon_eligibility.horizon_seconds == row.horizon_seconds)
            & (horizon_eligibility.cohort == "common_three_target")
        ].iloc[0]
        added_percent = 100.0 * float(row.added_origins_vs_common) / float(
            common.origins
        )
        expected = {
            "origin_support": (
                f"{int(common.origins):,}|{int(row.origins):,}|"
                f"{int(row.added_origins_vs_common):,} ({added_percent:.1f}%)"
            ),
            "mae": (
                f"{float(common.hierarchical_mae_bpm):.3f}|"
                f"{float(row.hierarchical_mae_bpm):.3f}"
            ),
            "delta": (
                f"{float(row.mae_delta_vs_common_bpm):.3f} "
                f"[{float(row.delta_ci_lower_bpm):.3f}, "
                f"{float(row.delta_ci_upper_bpm):.3f}]"
            ),
        }
        for metric, token in expected.items():
            if metric in {"origin_support", "mae"}:
                parts = token.split("|")
                ok = all(part in supplement for part in parts)
            else:
                ok = token in supplement
            key = f"v029_eligibility::{row.regime}::{int(row.horizon_seconds)}::{metric}"
            record(
                key,
                ok,
                f"Missing v0.29 eligibility token for {key}: {token}",
            )
    for _, row in horizon_frozen.iterrows():
        expected = {
            "common_mae": (
                f"{float(row.common_mae_median_bpm):.3f} "
                f"[{float(row.common_mae_min_bpm):.3f} to "
                f"{float(row.common_mae_max_bpm):.3f}]"
            ),
            "expanded_mae": (
                f"{float(row.horizon_specific_mae_median_bpm):.3f} "
                f"[{float(row.horizon_specific_mae_min_bpm):.3f} to "
                f"{float(row.horizon_specific_mae_max_bpm):.3f}]"
            ),
            "seed_delta": (
                f"{float(row.expanded_minus_common_seed_median_bpm):.3f} "
                f"[{float(row.expanded_minus_common_seed_min_bpm):.3f} to "
                f"{float(row.expanded_minus_common_seed_max_bpm):.3f}]"
            ),
            "paired_delta": (
                f"{float(row.paired_user_estimate_bpm):.3f} "
                f"[{float(row.paired_user_ci_lower_bpm):.3f}, "
                f"{float(row.paired_user_ci_upper_bpm):.3f}]"
            ),
            "added_origins": f"{int(row.added_origins):,}",
        }
        for metric, token in expected.items():
            key = (
                f"v030_frozen_eligibility::{row.regime}::"
                f"{int(row.horizon_seconds)}::{metric}"
            )
            record(
                key,
                token in supplement,
                f"Missing v0.30 frozen eligibility token for {key}: {token}",
            )

    for token in (
        "-0.013 to 0.155 bpm",
        "-0.010 to 0.154 bpm",
        "at most 0.155 bpm",
    ):
        record(
            f"main_v030::{token}",
            token in main_text,
            f"Missing v0.30 main-text summary token: {token}",
        )
    record(
        "main_v030::percentage_range_twice",
        main_text.count("2.7%–13.9%") == 2,
        "The corrected 2.7%–13.9% range must appear once in Results and once in Limitations",
    )

    balanced_90 = balanced_calibration[
        balanced_calibration.nominal_coverage == 0.9
    ]
    for _, row in balanced_90.iterrows():
        token = (
            f"{float(row.picp_median):.3f} "
            f"[{float(row.picp_min):.3f} to {float(row.picp_max):.3f}]"
        )
        key = (
            f"v024_balanced::{row.regime}::{row['mode']}::"
            f"{row.calibration_method}::{int(row.horizon_seconds)}"
        )
        record(
            key,
            token in supplement,
            f"Missing v0.24 calibration token for {key}: {token}",
        )

    interval_specs = [
        ("temporal_main", "within_user_temporal_test", "history_informed"),
        ("unseen_main", "unseen_user_test", "zero_history"),
        ("unseen_main", "goldencheetah_frozen_external", "zero_history"),
    ]
    for experiment, regime, mode in interval_specs:
        for metric, digits in [("picp", 3), ("mean_interval_width_bpm", 2)]:
            selected = q1_metric(
                q1,
                experiment=experiment,
                regime=regime,
                mode=mode,
                metric=metric,
                source_kind="interval",
                nominal=0.9,
                calibrated=True,
            )
            for _, row in selected.iterrows():
                token = sup_range(row, digits)
                key = f"interval::{regime}::{metric}::{int(row.horizon_seconds)}"
                record(key, token in supplement, f"Missing interval token for {key}: {token}")

    sport_point = q1[
        (q1.experiment == "held_sport")
        & (q1.source_kind == "point")
        & (q1.metric == "mae_bpm")
        & (q1["mode"] == "history_informed")
    ]
    for _, row in sport_point.iterrows():
        token = sup_range(row, 3)
        key = f"sport_point::{row.regime}::{int(row.horizon_seconds)}"
        record(key, token in supplement, f"Missing held-sport point token for {key}: {token}")

    sport_interval = q1[
        (q1.experiment == "held_sport")
        & (q1.source_kind == "interval")
        & (q1["mode"] == "history_informed")
        & (q1.calibrated == True)
        & (q1.nominal_coverage == 0.9)
        & (q1.metric.isin(["picp", "mean_interval_width_bpm"]))
    ]
    for _, row in sport_interval.iterrows():
        digits = 3 if row.metric == "picp" else 2
        token = sup_range(row, digits)
        key = f"sport_interval::{row.regime}::{row.metric}::{int(row.horizon_seconds)}"
        record(key, token in supplement, f"Missing held-sport interval token for {key}: {token}")

    history_effect = pd.read_csv(Q1 / "main_history_difference_summary_v0_22_0.csv")
    history_effect = history_effect[
        history_effect.experiment.isin(["temporal_main", "unseen_main"])
        & history_effect.regime.isin(
            ["within_user_temporal_test", "unseen_user_test"]
        )
    ]
    comparator_effect = pd.read_csv(Q1 / "main_vs_comparator_summary_v0_22_0.csv")
    comparator_effect = comparator_effect[
        comparator_effect.regime.isin(
            [
                "within_user_temporal_test",
                "unseen_user_test",
                "goldencheetah_frozen_external",
            ]
        )
    ]
    for family, frame in [
        ("history", history_effect),
        ("comparator", comparator_effect),
    ]:
        for _, row in frame.iterrows():
            token = (
                f"{float(row.difference_median_bpm):.3f} "
                f"[{float(row.difference_minimum_bpm):.3f} to "
                f"{float(row.difference_maximum_bpm):.3f}]"
            )
            key = f"seed_effect::{family}::{row.regime}::{int(row.horizon_seconds)}"
            record(key, token in supplement, f"Missing seed-effect token for {key}: {token}")

    history = read_result("history_availability_v0_19_0.csv")
    for _, row in history.iterrows():
        token = (
            f"{float(row.prior_count_session_median):.1f} "
            f"[{float(row.prior_count_session_q1):.1f}, "
            f"{float(row.prior_count_session_q3):.1f}]"
        )
        key = f"history_availability::{row.regime}"
        record(key, token in supplement, f"Missing history-availability token: {token}")

    for number in range(1, 20):
        token = f"## Table S{number}."
        record(
            f"supplement_heading::{number}",
            token in supplement,
            f"Missing supplementary table heading: {token}",
        )
    record(
        "supplement_v0_22_authority",
        "v0.22 aggregation" in supplement,
        "Supplement lacks the v0.22 authority statement",
    )
    record(
        "supplement_v0_23_authority",
        "v0.23 zero-history-trained aggregation" in supplement,
        "Supplement lacks the v0.23 authority statement",
    )
    record(
        "supplement_v0_24_authority",
        "Frozen-prediction v0.24 analyses" in supplement,
        "Supplement lacks the v0.24 authority statement",
    )
    record(
        "supplement_v0_25_authority",
        "multiseed_paired_model_comparisons_v0_25_0.csv" in supplement,
        "Supplement lacks the v0.25 authority statement",
    )
    record(
        "supplement_v0_26_authority",
        "v0.26 provides an independently calibrated persistence interval baseline"
        in supplement,
        "Supplement lacks the v0.26 authority statement",
    )
    record(
        "supplement_v0_27_authority",
        "v0.27 provides a post hoc matched-origin sport-availability sensitivity"
        in supplement,
        "Supplement lacks the v0.27 authority statement",
    )
    record(
        "supplement_v0_28_authority",
        "Version 0.28 is a separately trained, deliberately invalid negative control"
        in supplement,
        "Supplement lacks the v0.28 authority statement",
    )
    record(
        "supplement_v0_29_authority",
        "Version 0.29 provides a parameter-free, fixed-session target-availability sensitivity"
        in supplement,
        "Supplement lacks the v0.29 authority statement",
    )
    record(
        "supplement_v0_30_authority",
        "Version 0.30 applies the five frozen main-model seeds" in supplement,
        "Supplement lacks the v0.30 authority statement",
    )
    record(
        "supplement_no_missing_tokens",
        not bool(re.search(r"\b(?:nan|None)\b", supplement, flags=re.IGNORECASE)),
        "Supplement contains a missing-value token",
    )

    provenance = supplement.partition("## Supplementary provenance")[2]
    provenance_paths = sorted(
        set(re.findall(r"`([^`]+\.(?:csv|json|md))`", provenance, flags=re.IGNORECASE))
    )
    for token in provenance_paths:
        normalized = token.replace("\\", "/")
        if "/" in normalized:
            candidates = [ROOT / Path(normalized)]
        else:
            candidates = [
                RESULTS / normalized,
                Q1 / normalized,
                ZERO / normalized,
                LEAKY / normalized,
                ROOT / "outputs" / "audit" / normalized,
                ROOT / "references" / normalized,
            ]
        matches = [candidate for candidate in candidates if candidate.is_file()]
        record(
            f"provenance_path::{normalized}",
            len(matches) == 1,
            f"Supplementary provenance path is missing or ambiguous: {token}",
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "authoritative_versions": {
            "multiseed": "0.22.0",
            "independent_zero_history": "0.23.0",
            "balanced_calibration_and_interval_standardization": "0.24.0",
            "paired_user_bootstrap": "0.25.0",
            "independent_persistence_conformal_baseline": "0.26.0",
            "matched_origin_sport_availability": "0.27.0",
            "deliberately_leaky_negative_control": "0.28.0",
            "horizon_specific_target_eligibility": "0.29.0",
            "horizon_specific_frozen_models": "0.30.0",
            "reference_seed": 20260722,
        },
        "check_count": len(checks),
        "checks_passed": sum(checks.values()),
        "checks": checks,
        "errors": errors,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
