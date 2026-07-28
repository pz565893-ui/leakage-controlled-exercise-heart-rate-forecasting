"""Build evidence-linked supplementary tables from final result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
AUDIT = ROOT / "outputs" / "audit"
Q1_AGGREGATION = ROOT / "outputs" / "q1_multiseed_v0_21_0" / "aggregation"
ZERO_AGGREGATION = (
    ROOT / "outputs" / "independent_zero_history_v0_23_0" / "aggregation"
)
LEAKY_AGGREGATION = (
    ROOT / "outputs" / "deliberately_leaky_negative_control_v0_28_0" / "aggregation"
)
OUTPUT = ROOT / "manuscript" / "supplementary_material.md"

HORIZON_LABELS = {60: "1 min", 180: "3 min", 300: "5 min"}
REGIME_LABELS = {
    "within_user_temporal_test": "Strict temporal",
    "unseen_user_test": "Unseen user",
    "goldencheetah_frozen_external": "Frozen cross-source",
}
MODE_LABELS = {
    "history_informed": "History-informed",
    "history_masked": "History-masked",
    "zero_history": "History-masked",
    "not_applicable": "Not applicable",
}


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def read_json(name: str):
    return json.loads((AUDIT / name).read_text(encoding="utf-8"))


def read_q1(name: str) -> pd.DataFrame:
    return pd.read_csv(Q1_AGGREGATION / name)


def read_zero(name: str) -> pd.DataFrame:
    return pd.read_csv(ZERO_AGGREGATION / name)


def fmt_num(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def fmt_seed_range(median, minimum, maximum, digits: int = 3) -> str:
    if pd.isna(median) or pd.isna(minimum) or pd.isna(maximum):
        return "—"
    return (
        f"{float(median):.{digits}f} "
        f"[{float(minimum):.{digits}f} to {float(maximum):.{digits}f}]"
    )


def fmt_int(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(value):,}"


def fmt_p(value) -> str:
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def escape(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(escape(row.get(key, "")) for key, _ in columns) + " |"
        )
    return "\n".join(lines)


def point_wide(frame: pd.DataFrame, group_columns: list[str]) -> list[dict]:
    rows: list[dict] = []
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for _, item in group.sort_values("horizon_seconds").iterrows():
            label = HORIZON_LABELS[int(item.horizon_seconds)]
            row[f"{label} MAE"] = fmt_num(item.mae_bpm)
            row[f"{label} RMSE"] = fmt_num(item.rmse_bpm)
            row[f"{label} bias"] = fmt_num(item.bias_bpm)
        first = group.iloc[0]
        row["users"] = fmt_int(first.users)
        row["sessions"] = fmt_int(first.sessions)
        row["origins"] = fmt_int(first.origins)
        rows.append(row)
    return rows


def point_columns(label_key: str, label: str) -> list[tuple[str, str]]:
    columns = [(label_key, label)]
    for horizon in ("1 min", "3 min", "5 min"):
        columns.extend(
            [
                (f"{horizon} MAE", f"{horizon} MAE"),
                (f"{horizon} RMSE", f"{horizon} RMSE"),
                (f"{horizon} bias", f"{horizon} bias"),
            ]
        )
    columns.extend([("users", "Users"), ("sessions", "Sessions"), ("origins", "Origins")])
    return columns


def build_dataset_flow() -> tuple[str, str]:
    origins = read_json("forecast_origins_full_v0_3_1.json")
    split = read_json("split_manifest_v0_2_0_summary.json")
    users = pd.read_csv(ROOT / "outputs" / "features" / "model_arrays_v0_6_0" / "users.csv")
    user_counts = users.groupby("dataset").user_index.nunique().to_dict()
    processed = {item["dataset"]: item for item in origins["processed_sessions"]}
    origin_counts = {item["dataset"]: item for item in origins["origins_by_dataset"]}
    rows = []
    for dataset in ("Endomondo", "GoldenCheetah"):
        rows.append(
            {
                "dataset": dataset,
                "sessions": fmt_int(processed[dataset]["sessions"]),
                "users": fmt_int(user_counts[dataset]),
                "candidates": fmt_int(processed[dataset]["candidate_origins"]),
                "accepted": fmt_int(processed[dataset]["accepted_origins"]),
                "rate": fmt_num(
                    processed[dataset]["accepted_origins"]
                    / processed[dataset]["candidate_origins"]
                    * 100,
                    1,
                ),
                "primary": fmt_int(origin_counts[dataset]["evaluation_origins"]),
            }
        )
    table_a = markdown_table(
        rows,
        [
            ("dataset", "Dataset"),
            ("sessions", "Sessions entering origin construction"),
            ("users", "Users with accepted origins"),
            ("candidates", "Candidate origins"),
            ("accepted", "Accepted origins"),
            ("rate", "Accepted (%)"),
            ("primary", "Complete 300-s origin pool"),
        ],
    )

    endo_users = split["endomondo"]["unseen_user_partition_users"]
    support_rows = [
        {"regime": "Unseen-user training", "users": fmt_int(endo_users["train"]), "boundary": "User disjoint"},
        {"regime": "Unseen-user validation", "users": fmt_int(endo_users["validation"]), "boundary": "User disjoint"},
        {"regime": "Unseen-user calibration", "users": fmt_int(endo_users["calibration"]), "boundary": "User disjoint"},
        {"regime": "Unseen-user test", "users": fmt_int(endo_users["test"]), "boundary": "User disjoint"},
        {"regime": "Strict temporal test", "users": "948", "boundary": "Later sessions; crossing sessions excluded"},
        {"regime": "Frozen GoldenCheetah cross-source", "users": "144", "boundary": "Three shared sport families; 531,725 origins; 31,851 sessions; history-masked; no adaptation or recalibration"},
    ]
    table_b = markdown_table(
        support_rows,
        [("regime", "Partition/regime"), ("users", "Users"), ("boundary", "Primary boundary")],
    )
    return table_a, table_b


def mae_summary_columns(label_key: str, label: str) -> list[tuple[str, str]]:
    return [
        (label_key, label),
        ("1 min", "1-min MAE, bpm"),
        ("3 min", "3-min MAE, bpm"),
        ("5 min", "5-min MAE, bpm"),
        ("seeds", "Runs/seeds"),
        ("users", "Users"),
        ("sessions", "Sessions"),
        ("origins", "Origins"),
    ]


def q1_point_summary_row(
    summary: pd.DataFrame,
    *,
    experiment: str,
    regime: str,
    mode: str,
    label: str,
) -> dict[str, str]:
    selected = summary[
        (summary.experiment == experiment)
        & (summary.source_kind == "point")
        & (summary.regime == regime)
        & (summary["mode"] == mode)
        & (summary.metric == "mae_bpm")
    ].copy()
    if len(selected) != 3:
        raise AssertionError(f"expected three Q1 point rows for {label}, found {len(selected)}")
    row: dict[str, str] = {"label": label}
    for _, item in selected.sort_values("horizon_seconds").iterrows():
        row[HORIZON_LABELS[int(item.horizon_seconds)]] = fmt_seed_range(
            item.value_median, item.value_minimum, item.value_maximum
        )
    first = selected.iloc[0]
    row.update(
        {
            "seeds": str(int(first.n_seeds)),
            "users": fmt_int(first.users),
            "sessions": fmt_int(first.sessions),
            "origins": fmt_int(first.origins),
        }
    )
    return row


def independent_point_summary_row(
    per_seed: pd.DataFrame,
    *,
    protocol: str,
    evaluation: str,
    label: str,
) -> dict[str, str]:
    selected = per_seed[
        (per_seed.protocol == protocol) & (per_seed.evaluation == evaluation)
    ][
        [
            "seed",
            "horizon_seconds",
            "independent_zero_mae_bpm",
            "users",
            "sessions",
            "origins",
        ]
    ].drop_duplicates()
    if len(selected) != 15:
        raise AssertionError(
            f"expected fifteen independent-zero point rows for {label}, found {len(selected)}"
        )
    row: dict[str, str] = {"label": label}
    for horizon, group in selected.groupby("horizon_seconds", sort=True):
        values = group.independent_zero_mae_bpm.to_numpy(dtype=float)
        row[HORIZON_LABELS[int(horizon)]] = fmt_seed_range(
            np.median(values), np.min(values), np.max(values)
        )
    first = selected.iloc[0]
    row.update(
        {
            "seeds": str(selected.seed.nunique()),
            "users": fmt_int(first.users),
            "sessions": fmt_int(first.sessions),
            "origins": fmt_int(first.origins),
        }
    )
    return row


def deterministic_point_rows(
    frame: pd.DataFrame,
    *,
    regime: str,
    model_labels: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    selected = frame[frame.regime == regime] if "regime" in frame else frame
    for model, label in model_labels.items():
        group = selected[selected.model == model].sort_values("horizon_seconds")
        if len(group) != 3:
            raise AssertionError(f"expected three deterministic rows for {regime}/{model}")
        row: dict[str, str] = {"label": label, "seeds": "Deterministic"}
        for _, item in group.iterrows():
            row[HORIZON_LABELS[int(item.horizon_seconds)]] = fmt_num(item.mae_bpm)
        first = group.iloc[0]
        row.update(
            {
                "users": fmt_int(first.users),
                "sessions": fmt_int(first.sessions),
                "origins": fmt_int(first.origins),
            }
        )
        rows.append(row)
    return rows


def single_run_point_row(
    frame: pd.DataFrame, *, regime: str, label: str
) -> dict[str, str]:
    selected = frame[frame.regime == regime].sort_values("horizon_seconds")
    if len(selected) != 3:
        raise AssertionError(f"expected three single-run rows for {label}/{regime}")
    row: dict[str, str] = {"label": label, "seeds": "1 (20260722)"}
    for _, item in selected.iterrows():
        row[HORIZON_LABELS[int(item.horizon_seconds)]] = fmt_num(item.mae_bpm)
    first = selected.iloc[0]
    row.update(
        {
            "users": fmt_int(first.users),
            "sessions": fmt_int(first.sessions),
            "origins": fmt_int(first.origins),
        }
    )
    return row


def build_temporal_point() -> str:
    summary = read_q1("seed_variability_summary_v0_22_0.csv")
    independent = read_zero("strategy_contrasts_per_seed_v0_23_0.csv")
    rows = [
        q1_point_summary_row(
            summary,
            experiment="temporal_main",
            regime="within_user_temporal_test",
            mode="history_informed",
            label="History-quantile TCN (history)",
        ),
        q1_point_summary_row(
            summary,
            experiment="temporal_main",
            regime="within_user_temporal_test",
            mode="zero_history",
            label="History-capable TCN (history-masked)",
        ),
        independent_point_summary_row(
            independent,
            protocol="strict_temporal",
            evaluation="internal_test",
            label="Zero-history-trained TCN",
        ),
        q1_point_summary_row(
            summary,
            experiment="temporal_gru",
            regime="within_user_temporal_test",
            mode="not_applicable",
            label="GRU",
        ),
        q1_point_summary_row(
            summary,
            experiment="temporal_tcn",
            regime="within_user_temporal_test",
            mode="not_applicable",
            label="Point TCN",
        ),
    ]
    base = read_csv("temporal_aligned_baselines_v0_13_0.csv").copy()
    base["regime"] = "within_user_temporal_test"
    rows.extend(
        deterministic_point_rows(
            base,
            regime="within_user_temporal_test",
            model_labels={
                "persistence": "Persistence",
                "ewma_alpha_0_1": "EWMA",
                "linear_trend": "Linear trend",
            },
        )
    )
    return markdown_table(rows, mae_summary_columns("label", "Model"))


def build_user_external_point() -> str:
    summary = read_q1("seed_variability_summary_v0_22_0.csv")
    independent = read_zero("strategy_contrasts_per_seed_v0_23_0.csv")
    naive = read_csv("naive_baseline_metrics_v0_5_0.csv")
    single_frames = {
        "XGBoost": read_csv("xgboost_user_generalization_metrics_v0_8_0.csv"),
        "Transformer": read_csv("transformer_user_generalization_metrics_v0_9_0.csv"),
    }
    rows: list[dict[str, str]] = []
    for regime, experiment_label, evaluation in [
        ("unseen_user_test", "Unseen user", "internal_test"),
        ("goldencheetah_frozen_external", "Frozen cross-source", "frozen_external"),
    ]:
        if regime == "unseen_user_test":
            rows.append(
                q1_point_summary_row(
                    summary,
                    experiment="unseen_main",
                    regime=regime,
                    mode="history_informed",
                    label=f"{experiment_label}: history-quantile TCN (history)",
                )
            )
        rows.append(
            q1_point_summary_row(
                summary,
                experiment="unseen_main",
                regime=regime,
                mode="zero_history",
                label=f"{experiment_label}: history-capable TCN (history-masked)",
            )
        )
        rows.append(
            independent_point_summary_row(
                independent,
                protocol="unseen_user",
                evaluation=evaluation,
                label=f"{experiment_label}: zero-history-trained TCN",
            )
        )
        for experiment, label in [("unseen_gru", "GRU"), ("unseen_tcn", "Point TCN")]:
            rows.append(
                q1_point_summary_row(
                    summary,
                    experiment=experiment,
                    regime=regime,
                    mode="not_applicable",
                    label=f"{experiment_label}: {label}",
                )
            )
        for label, frame in single_frames.items():
            rows.append(
                single_run_point_row(
                    frame,
                    regime=regime,
                    label=f"{experiment_label}: {label}",
                )
            )
        deterministic = deterministic_point_rows(
            naive,
            regime=regime,
            model_labels={
                "persistence": "Persistence",
                "ewma": "EWMA",
                "linear_trend": "Linear trend",
            },
        )
        for row in deterministic:
            row["label"] = f"{experiment_label}: {row['label']}"
        rows.extend(deterministic)
    return markdown_table(rows, mae_summary_columns("label", "Regime and model"))


def build_uncertainty() -> str:
    summary = read_q1("seed_variability_summary_v0_22_0.csv")
    summary["calibrated_bool"] = (
        summary.calibrated.astype(str).str.lower().map({"true": True, "false": False})
    )
    specifications = [
        ("temporal_main", "within_user_temporal_test", "history_informed"),
        ("unseen_main", "unseen_user_test", "history_informed"),
        ("unseen_main", "goldencheetah_frozen_external", "zero_history"),
    ]
    rows: list[dict[str, str]] = []
    for experiment, regime, mode in specifications:
        selected = summary[
            (summary.experiment == experiment)
            & (summary.source_kind == "interval")
            & (summary.regime == regime)
            & (summary["mode"] == mode)
        ].copy()
        for nominal in sorted(selected.nominal_coverage.dropna().unique()):
            for calibrated in (False, True):
                row: dict[str, str] = {
                    "regime": REGIME_LABELS[regime],
                    "mode": MODE_LABELS.get(mode, mode),
                    "nominal": f"{int(round(float(nominal) * 100))}%",
                    "calibration": "CQR" if calibrated else "Raw quantiles",
                    "seeds": "5",
                }
                for horizon in HORIZON_LABELS:
                    for metric, suffix, digits in [
                        ("picp", "PICP", 3),
                        ("mean_interval_width_bpm", "width", 2),
                        ("conformal_adjustment_bpm", "adj", 3),
                    ]:
                        metric_row = selected[
                            np.isclose(selected.nominal_coverage, nominal)
                            & (selected.calibrated_bool == calibrated)
                            & (selected.horizon_seconds == horizon)
                            & (selected.metric == metric)
                        ]
                        key = f"{HORIZON_LABELS[horizon]} {suffix}"
                        if len(metric_row) == 1:
                            item = metric_row.iloc[0]
                            row[key] = fmt_seed_range(
                                item.value_median,
                                item.value_minimum,
                                item.value_maximum,
                                digits,
                            )
                        elif metric == "conformal_adjustment_bpm" and not calibrated:
                            row[key] = "—"
                        else:
                            raise AssertionError(
                                f"unexpected interval summary cardinality for "
                                f"{regime}/{nominal}/{calibrated}/{horizon}/{metric}: "
                                f"{len(metric_row)}"
                            )
                rows.append(row)
    columns = [
        ("regime", "Regime"),
        ("mode", "Mode"),
        ("nominal", "Nominal"),
        ("calibration", "Intervals"),
    ]
    for horizon in ("1 min", "3 min", "5 min"):
        columns.extend(
            [
                (f"{horizon} PICP", f"{horizon} PICP"),
                (f"{horizon} width", f"{horizon} width"),
                (f"{horizon} adj", f"{horizon} CQR adj."),
            ]
        )
    columns.append(("seeds", "Seeds"))
    return markdown_table(rows, columns)


def build_figure3_uncertainty() -> str:
    frame = read_csv("figure3_uncertainty_bootstrap_v0_18_0.csv")
    rows = []
    for _, item in frame.iterrows():
        rows.append(
            {
                "regime": REGIME_LABELS.get(item.regime, item.regime),
                "mode": MODE_LABELS.get(item["mode"], item["mode"]),
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "picp": (
                    f"{float(item.picp):.3f} "
                    f"[{float(item.picp_ci_low):.3f}, {float(item.picp_ci_high):.3f}]"
                ),
                "width": (
                    f"{float(item.mean_90_interval_width_bpm):.2f} "
                    f"[{float(item.mean_90_interval_width_ci_low):.2f}, "
                    f"{float(item.mean_90_interval_width_ci_high):.2f}]"
                ),
                "wis": (
                    f"{float(item.weighted_interval_score):.2f} "
                    f"[{float(item.weighted_interval_score_ci_low):.2f}, "
                    f"{float(item.weighted_interval_score_ci_high):.2f}]"
                ),
                "spearman": (
                    f"{float(item.mean_user_spearman_width_absolute_error):.3f} "
                    f"[{float(item.mean_user_spearman_ci_low):.3f}, "
                    f"{float(item.mean_user_spearman_ci_high):.3f}]"
                ),
                "users": fmt_int(item.users),
            }
        )
    return markdown_table(
        rows,
        [
            ("regime", "Regime"),
            ("mode", "Mode"),
            ("horizon", "Horizon"),
            ("picp", "90% PICP [95% CI]"),
            ("width", "90% width [95% CI], bpm"),
            ("wis", "WIS [95% CI]"),
            ("spearman", "Width-error Spearman [95% CI]"),
            ("users", "Users"),
        ],
    )


def build_sport_shift() -> str:
    frame = read_q1("seed_variability_summary_v0_22_0.csv")
    frame = frame[
        (frame.experiment == "held_sport")
        & (frame.source_kind == "point")
        & (frame["mode"] == "history_informed")
        & (frame.metric == "mae_bpm")
    ].copy()
    frame["regime_label"] = np.where(
        frame.regime.str.startswith("joint_user_sport"),
        "Joint user–sport",
        "Same-user unseen sport",
    )
    frame["sport_label"] = frame.family.str.replace("_", " ", regex=False)
    rows: list[dict[str, str]] = []
    for (regime, sport), group in frame.groupby(
        ["regime_label", "sport_label"], sort=False
    ):
        row: dict[str, str] = {"regime": regime, "sport": sport, "seeds": "3"}
        for _, item in group.sort_values("horizon_seconds").iterrows():
            row[HORIZON_LABELS[int(item.horizon_seconds)]] = fmt_seed_range(
                item.value_median, item.value_minimum, item.value_maximum
            )
        first = group.iloc[0]
        row["users"] = fmt_int(first.users)
        row["sessions"] = fmt_int(first.sessions)
        row["origins"] = fmt_int(first.origins)
        row["support"] = (
            "Caution (<25 users)"
            if int(first.users) < 25
            else "Supported descriptive"
        )
        rows.append(row)
    return markdown_table(
        rows,
        [
            ("regime", "Regime"),
            ("sport", "Held sport family"),
            ("1 min", "1-min MAE [seed range]"),
            ("3 min", "3-min MAE [seed range]"),
            ("5 min", "5-min MAE [seed range]"),
            ("seeds", "Seeds"),
            ("users", "Users"),
            ("sessions", "Sessions"),
            ("origins", "Origins"),
            ("support", "Support interpretation"),
        ],
    )


def build_sport_uncertainty() -> str:
    frame = read_q1("seed_variability_summary_v0_22_0.csv")
    frame["calibrated_bool"] = frame.calibrated.astype(str).str.lower() == "true"
    frame = frame[
        (frame.experiment == "held_sport")
        & (frame.source_kind == "interval")
        & (frame["mode"] == "history_informed")
        & (frame.calibrated_bool)
        & np.isclose(frame.nominal_coverage, 0.9)
        & frame.metric.isin(["picp", "mean_interval_width_bpm"])
    ].copy()
    frame["regime_label"] = np.where(
        frame.regime.str.startswith("joint_user_sport"),
        "Joint user–sport",
        "Same-user unseen sport",
    )
    frame["sport_label"] = frame.family.str.replace("_", " ", regex=False)
    rows: list[dict[str, str]] = []
    for (regime, sport), group in frame.groupby(
        ["regime_label", "sport_label"], sort=False
    ):
        row: dict[str, str] = {"regime": regime, "sport": sport, "seeds": "3"}
        for horizon in HORIZON_LABELS:
            for metric, suffix, digits in [
                ("picp", "PICP", 3),
                ("mean_interval_width_bpm", "width", 2),
            ]:
                item = group[
                    (group.horizon_seconds == horizon) & (group.metric == metric)
                ]
                if len(item) != 1:
                    raise AssertionError(
                        f"unexpected held-sport interval cardinality: "
                        f"{regime}/{sport}/{horizon}/{metric}"
                    )
                value = item.iloc[0]
                row[f"{HORIZON_LABELS[horizon]} {suffix}"] = fmt_seed_range(
                    value.value_median,
                    value.value_minimum,
                    value.value_maximum,
                    digits,
                )
        first = group.iloc[0]
        row["users"] = fmt_int(first.users)
        row["support"] = (
            "Caution (<25 users)"
            if int(first.users) < 25
            else "Supported descriptive"
        )
        rows.append(row)
    columns = [("regime", "Regime"), ("sport", "Held sport family")]
    for horizon in ("1 min", "3 min", "5 min"):
        columns.extend(
            [
                (f"{horizon} PICP", f"{horizon} PICP [seed range]"),
                (f"{horizon} width", f"{horizon} width [seed range]"),
            ]
        )
    columns.extend(
        [
            ("seeds", "Seeds"),
            ("users", "Users"),
            ("support", "Support interpretation"),
        ]
    )
    return markdown_table(rows, columns)


def build_seed_paired_effects() -> str:
    history = read_q1("main_history_difference_summary_v0_22_0.csv")
    history = history[
        history.experiment.isin(["temporal_main", "unseen_main"])
        & history.regime.isin(["within_user_temporal_test", "unseen_user_test"])
    ].copy()
    rows: list[dict[str, str]] = []
    for _, item in history.iterrows():
        rows.append(
            {
                "comparison": "History-informed − history-masked",
                "regime": REGIME_LABELS[item.regime],
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "effect": fmt_seed_range(
                    item.difference_median_bpm,
                    item.difference_minimum_bpm,
                    item.difference_maximum_bpm,
                ),
                "seeds": str(int(item.n_seed_pairs)),
            }
        )
    comparators = read_q1("main_vs_comparator_summary_v0_22_0.csv")
    comparators = comparators[
        comparators.regime.isin(
            [
                "within_user_temporal_test",
                "unseen_user_test",
                "goldencheetah_frozen_external",
            ]
        )
    ].copy()
    for _, item in comparators.iterrows():
        rows.append(
            {
                "comparison": f"History-masked − {str(item.comparator_model).upper()}",
                "regime": REGIME_LABELS[item.regime],
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "effect": fmt_seed_range(
                    item.difference_median_bpm,
                    item.difference_minimum_bpm,
                    item.difference_maximum_bpm,
                ),
                "seeds": str(int(item.n_seed_pairs)),
            }
        )
    return markdown_table(
        rows,
        [
            ("comparison", "Comparison (left minus right)"),
            ("regime", "Regime"),
            ("horizon", "Horizon"),
            ("effect", "Median ΔMAE [seed range], bpm"),
            ("seeds", "Matched seeds"),
        ],
    )


def build_sport_paired_bootstrap_v025() -> str:
    """Report three-seed paired-user main-minus-EWMA sport effects."""
    frame = read_csv("multiseed_paired_sport_shift_v0_25_0.csv")
    regime_labels = {
        "same_user_unseen_sport": "Same-user held sport",
        "joint_user_sport": "Joint user--sport (exploratory)",
    }
    rows = []
    for _, item in frame.sort_values(
        ["regime", "held_sport_family", "horizon_seconds"]
    ).iterrows():
        rows.append(
            {
                "regime": regime_labels.get(item.regime, item.regime),
                "sport": str(item.held_sport_family).replace("_", " "),
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "delta": (
                    f"{float(item.delta_mae_bpm):.3f} "
                    f"[{float(item.ci_low_bpm):.3f}, "
                    f"{float(item.ci_high_bpm):.3f}]"
                ),
                "users": fmt_int(item.users),
                "seeds": fmt_int(item.n_matched_seeds),
                "support": (
                    "Caution (<25 users)"
                    if str(item.joint_support_caution_lt25_users).lower() == "true"
                    else (
                        "Exploratory"
                        if item.regime == "joint_user_sport"
                        else "Supported descriptive"
                    )
                ),
            }
        )
    return markdown_table(
        rows,
        [
            ("regime", "Regime"),
            ("sport", "Held sport family"),
            ("horizon", "Horizon"),
            ("delta", "Main minus aligned EWMA MAE [95% CI], bpm"),
            ("users", "Users"),
            ("seeds", "Matched seeds"),
            ("support", "Interpretation"),
        ],
    )


def build_model_paired_bootstrap_v025() -> str:
    """Report three-seed paired-user main-minus-learned-comparator effects."""
    frame = read_csv("multiseed_paired_model_comparisons_v0_25_0.csv")
    regime_labels = {
        "strict_temporal": "Strict temporal",
        "unseen_user": "Unseen user",
        "goldencheetah": "Frozen cross-source",
        "goldencheetah_cross_source": "Frozen cross-source",
        "goldencheetah_frozen_external": "Frozen cross-source",
    }
    rows = []
    for _, item in frame.sort_values(
        ["regime", "comparator_model", "horizon_seconds"]
    ).iterrows():
        rows.append(
            {
                "regime": regime_labels.get(item.regime, item.regime),
                "mode": MODE_LABELS.get(item.main_mode, item.main_mode),
                "comparator": str(item.comparator_model).upper(),
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "main": fmt_num(item.main_mae_bpm, 3),
                "comparator_mae": fmt_num(item.comparator_mae_bpm, 3),
                "delta": (
                    f"{float(item.delta_mae_bpm):.3f} "
                    f"[{float(item.ci_low_bpm):.3f}, "
                    f"{float(item.ci_high_bpm):.3f}]"
                ),
                "users": fmt_int(item.users),
                "seeds": fmt_int(item.n_matched_seeds),
            }
        )
    return markdown_table(
        rows,
        [
            ("regime", "Regime"),
            ("mode", "Main-model mode"),
            ("comparator", "Comparator"),
            ("horizon", "Horizon"),
            ("main", "Main MAE, bpm"),
            ("comparator_mae", "Comparator MAE, bpm"),
            ("delta", "Main minus comparator MAE [95% CI], bpm"),
            ("users", "Users"),
            ("seeds", "Matched seeds"),
        ],
    )


def build_independent_strategy_effects() -> str:
    frame = read_zero("strategy_contrast_user_bootstrap_v0_23_0.csv")
    contrast_labels = {
        "mixed_history_minus_mixed_zero": "History-informed − history-masked",
        "mixed_zero_minus_independent_zero": "History-masked − zero-history-trained",
        "mixed_history_minus_independent_zero": "History-informed − zero-history-trained",
    }
    rows: list[dict[str, str]] = []
    for _, item in frame.iterrows():
        regime = (
            "Strict temporal"
            if item.protocol == "strict_temporal"
            else ("Frozen cross-source" if item.evaluation == "frozen_external" else "Unseen user")
        )
        low = float(item.percentile_95_ci_low_bpm)
        high = float(item.percentile_95_ci_high_bpm)
        rows.append(
            {
                "comparison": contrast_labels[item.contrast],
                "regime": regime,
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "effect": (
                    f"{float(item.estimate_bpm):.3f} [{low:.3f}, {high:.3f}]"
                ),
                "users": fmt_int(item.users),
                "seeds": str(int(item.matched_seeds_per_user)),
                "interpretation": "CI excludes 0" if high < 0 or low > 0 else "CI includes 0",
            }
        )
    return markdown_table(
        rows,
        [
            ("comparison", "Strategy contrast (left minus right)"),
            ("regime", "Regime"),
            ("horizon", "Horizon"),
            ("effect", "Seed-averaged user ΔMAE [95% CI], bpm"),
            ("users", "Users"),
            ("seeds", "Matched seeds/user"),
            ("interpretation", "Interpretation"),
        ],
    )


def build_reference_seed_effects() -> str:
    frame = read_csv("signal_ablation_paired_v0_14_0.csv")
    rows: list[dict[str, str]] = []
    for _, item in frame.iterrows():
        low = float(item.ci_low_bpm)
        high = float(item.ci_high_bpm)
        rows.append(
            {
                "family": item.comparison_family,
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "effect": f"{float(item.delta_mae_bpm):.3f} [{low:.3f}, {high:.3f}]",
                "p": fmt_p(item.holm_adjusted_wilcoxon_p_value),
                "users": fmt_int(item.users),
                "interpretation": "CI excludes 0" if high < 0 or low > 0 else "CI includes 0",
            }
        )
    return markdown_table(
        rows,
        [
            ("family", "Comparison family"),
            ("horizon", "Horizon"),
            ("effect", "ΔMAE [95% CI], bpm"),
            ("p", "Holm-adjusted Wilcoxon p"),
            ("users", "Users"),
            ("interpretation", "Reference-seed interpretation"),
        ],
    )


def build_stride() -> str:
    frame = pd.read_csv(ROOT / "figures" / "source_data" / "Supplementary_Figure_1_stride_source.csv")
    rows = []
    for _, item in frame.iterrows():
        rows.append(
            {
                "mode": MODE_LABELS.get(item["mode"], item["mode"]),
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "standard": fmt_num(item.standard),
                "dense": fmt_num(item.dense),
                "delta": fmt_num(item.delta),
            }
        )
    return markdown_table(
        rows,
        [
            ("mode", "Mode"),
            ("horizon", "Horizon"),
            ("standard", "300-s-origin MAE"),
            ("dense", "60-s-origin MAE"),
            ("delta", "Dense minus standard"),
        ],
    )


def build_gender() -> str:
    frame = read_csv("recorded_gender_differences_v0_16_0.csv")
    rows = []
    for _, item in frame.iterrows():
        rows.append(
            {
                "regime": REGIME_LABELS.get(item.regime, item.regime),
                "mode": MODE_LABELS.get(item["mode"], item["mode"]),
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "effect": f"{float(item.delta_mae_bpm):.3f} [{float(item.ci_low_bpm):.3f}, {float(item.ci_high_bpm):.3f}]",
                "support": f"{int(item.female_users)} recorded female / {int(item.male_users)} recorded male",
                "status": item.support_status.replace("_", " "),
            }
        )
    return markdown_table(
        rows,
        [
            ("regime", "Regime"),
            ("mode", "Mode"),
            ("horizon", "Horizon"),
            ("effect", "Female minus male MAE [95% CI], bpm"),
            ("support", "User support"),
            ("status", "Status"),
        ],
    )


def build_prior_work() -> str:
    rows = [
        {
            "study": "Present study",
            "task": "Future recorded exercise-HR forecasting",
            "range": "+1/+3/+5 min",
            "user": "Strict temporal + unseen-user",
            "history": "Completed prior workouts",
            "sport": "Five held families + joint user-sport shift",
            "external": "Frozen Endomondo-to-GoldenCheetah cross-source evaluation",
            "interval": "50/80/90% empirical post-CQR intervals",
        },
        {
            "study": "Qiu et al., 2021",
            "task": "Personalized mountain-biking HR forecasting",
            "range": "Future HR along one course; physical lead NR",
            "user": "No; one cyclist, chronological first-80/last-20 split",
            "history": "Personalized course/ride data; completed-workout encoder no",
            "sport": "No; mountain biking on one course",
            "external": "No; one rider and course",
            "interval": "No calibrated PI documented",
        },
        {
            "study": "Gilbert et al., 2022",
            "task": "Biking HR forecasting with future gradient values",
            "range": "Up to 10 min; future course gradient is an input",
            "user": "Participant-independent boundary NR",
            "history": "Current ride plus future route information",
            "sport": "No; biking only",
            "external": "No frozen source transfer documented",
            "interval": "No calibrated PI documented",
        },
        {
            "study": "Fedorin et al., 2021",
            "task": "HIIT HR-trend forecasting from consumer wearables",
            "range": "Future trend; exact physical lead NR from inspected record",
            "user": "Participant split NR",
            "history": "Completed-workout history NR",
            "sport": "No held-sport protocol documented",
            "external": "No frozen source transfer documented",
            "interval": "Predictive interval/coverage NR",
        },
        {
            "study": "Ni et al., 2019",
            "task": "Personalized Endomondo speed/HR modelling",
            "range": "Full profile; next 10-s sample",
            "user": "Within-user chronological; not unseen-user",
            "history": "User representation + most recent workout",
            "sport": "No held-sport test documented",
            "external": "No independent source",
            "interval": "No documented interval",
        },
        {
            "study": "Nazaret et al., 2023",
            "task": "Future-run HR profile modelling",
            "range": "Whole run, up to 2 h",
            "user": "Within-person temporal; not unseen-user",
            "history": "Prior-workout history encoder",
            "sport": "No; outdoor running only",
            "external": "No independent source documented",
            "interval": "No; fixed +/-5-bpm band is not a PI",
        },
        {
            "study": "Hallgrímsson et al., 2018",
            "task": "Minute-level free-living HR modelling",
            "range": "Future lead time NR",
            "user": "Same-person 2017-to-2018; not unseen-user",
            "history": "Longitudinal participant signature",
            "sport": "No held-sport test documented",
            "external": "No frozen HR-transfer dataset",
            "interval": "NR",
        },
        {
            "study": "Pacheco et al., 2024",
            "task": "Exercise HR estimation from accelerometry/demographics",
            "range": "Several minutes; exact lead definition NR",
            "user": "NR",
            "history": "Online within-workout adaptation",
            "sport": "NR",
            "external": "Five datasets; frozen source transfer NR",
            "interval": "NR",
        },
        {
            "study": "Reiss et al., 2019",
            "task": "PPG/accelerometer HR estimation",
            "range": "Current 8-s window; not future forecasting",
            "user": "Leave-one-session-out; subject-independent",
            "history": "No personalization",
            "sport": "No held-activity transfer",
            "external": "Datasets analysed separately; no frozen transfer",
            "interval": "No",
        },
        {
            "study": "Kayange et al., 2024",
            "task": "Personalized whole-workout HR modelling on FitRec",
            "range": "Whole profile; fixed +1/+3/+5-min leads NR",
            "user": "User-grouped 80/20; no independent final test",
            "history": "Completed recent/past workouts",
            "sport": "NR; no held-sport test reported",
            "external": "No; FitRec only",
            "interval": "No calibrated PI; fixed +/-5-bpm band",
        },
        {
            "study": "Namazi et al., 2025",
            "task": "Multivariate sports-HR prediction from HR/BR/RR",
            "range": "Future epochs; physical lead time NR",
            "user": "80/20 split; split unit NR",
            "history": "Completed-session/user history NR",
            "sport": "NR; no held-sport protocol reported",
            "external": "No; one Sport Database source",
            "interval": "NR",
        },
        {
            "study": "De Sabbata & Simonini, 2025",
            "task": "Per-user ARIMA/random-walk HR forecasting",
            "range": "Next 1 min; 15-to-150-min rolling inputs",
            "user": "Per-user chronological; no unseen-user test",
            "history": "Per-user fitting + recent same-user window",
            "sport": "No",
            "external": "No; two datasets fitted separately per user",
            "interval": "No",
        },
        {
            "study": "Mateescu et al., 2025",
            "task": "Activity-conditioned Transformer/diffusion forecasting",
            "range": "L input steps to L future steps; L/time unit NR",
            "user": "65/15/20 split; split unit NR",
            "history": "No completed-session history; current context only",
            "sport": "No; activity-specific, not held-out",
            "external": "No; one private Fitbit cohort",
            "interval": "No; sample median only, no coverage",
        },
        {
            "study": "Zhang et al., 2026",
            "task": "HR(t) estimation from contemporaneous VO2(t)",
            "range": "0 s; not future forecasting",
            "user": "Participant-wise 80/20; no unseen-user test",
            "history": "Participant-specific fit; no workout-history encoder",
            "sport": "No",
            "external": "No; both sources used in development",
            "interval": "No",
        },
        {
            "study": "Namazi, 2022",
            "task": "Univariate running-HR forecasting",
            "range": "Past 1,500 s to next 30 s",
            "user": "Participant-disjoint split NR",
            "history": "Same-record past 1,500 s; not completed workouts",
            "sport": "No; running only",
            "external": "No; one source",
            "interval": "No; copula draws averaged to a point",
        },
        {
            "study": "Zhu et al., 2022",
            "task": "HR + wrist-inertial exercise-HR forecasting",
            "range": "+5/+7/+10/+15/+20/+25 s",
            "user": "Yes; nine-fold participant split",
            "history": "No; current 5.12-s sensor window only",
            "sport": "No; separate model per activity",
            "external": "NR; TicWatch test not established as frozen",
            "interval": "No",
        },
    ]
    return markdown_table(
        rows,
        [
            ("study", "Study"),
            ("task", "Task"),
            ("range", "Prediction range/target"),
            ("user", "User boundary"),
            ("history", "Individual history"),
            ("sport", "Held-sport evaluation"),
            ("external", "External-data design"),
            ("interval", "Predictive intervals"),
        ],
    )


def build_history_availability() -> str:
    frame = read_csv("history_availability_v0_19_0.csv")
    labels = {
        "strict_temporal_test": "Strict temporal test",
        "unseen_user_test": "Unseen-user test",
    }
    rows = []
    for _, item in frame.iterrows():
        rows.append(
            {
                "regime": labels.get(item.regime, item.regime),
                "users": fmt_int(item.users),
                "sessions": fmt_int(item.sessions),
                "available": f"{float(item.sessions_with_history_percent):.1f}%",
                "prior": (
                    f"{float(item.prior_count_session_median):.1f} "
                    f"[{float(item.prior_count_session_q1):.1f}, "
                    f"{float(item.prior_count_session_q3):.1f}]"
                ),
                "zero": fmt_int(item.sessions_prior_0),
                "one_four": fmt_int(item.sessions_prior_1_4),
                "five_nine": fmt_int(item.sessions_prior_5_9),
                "ten_plus": fmt_int(item.sessions_prior_10_plus),
                "any": fmt_int(item.users_with_any_history),
                "all": fmt_int(item.users_with_history_in_all_test_sessions),
            }
        )
    return markdown_table(
        rows,
        [
            ("regime", "Regime"),
            ("users", "Users"),
            ("sessions", "Sessions"),
            ("available", "Sessions with history"),
            ("prior", "Prior sessions, median [Q1, Q3]"),
            ("zero", "0 prior"),
            ("one_four", "1-4 prior"),
            ("five_nine", "5-9 prior"),
            ("ten_plus", "10+ prior"),
            ("any", "Users with any history"),
            ("all", "Users with history in all test sessions"),
        ],
    )


def build_user_balanced_calibration() -> str:
    """Compare five-seed origin-pooled and equal-user/session 90% calibration."""
    frame = read_csv("multiseed_balanced_calibration_summary_v0_24_0.csv")
    frame = frame[np.isclose(frame["nominal_coverage"], 0.90)].copy()
    method_labels = {
        "origin_pooled_finite_sample_cqr": "Origin-pooled CQR",
        "equal_user_equal_session_empirical": "Equal-user/session empirical",
    }
    frame["_method_order"] = frame["calibration_method"].map(
        {
            "origin_pooled_finite_sample_cqr": 0,
            "equal_user_equal_session_empirical": 1,
        }
    )
    rows = []
    for _, item in frame.sort_values(
        ["regime", "mode", "horizon_seconds", "_method_order"]
    ).iterrows():
        rows.append(
            {
                "regime": REGIME_LABELS.get(item.regime, item.regime),
                "mode": MODE_LABELS.get(item["mode"], item["mode"]),
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "method": method_labels.get(item.calibration_method, item.calibration_method),
                "adjustment": fmt_seed_range(
                    item.conformal_adjustment_bpm_median,
                    item.conformal_adjustment_bpm_min,
                    item.conformal_adjustment_bpm_max,
                    3,
                ),
                "picp": fmt_seed_range(
                    item.picp_median, item.picp_min, item.picp_max, 3
                ),
                "error": fmt_seed_range(
                    item.absolute_coverage_error_median,
                    item.absolute_coverage_error_min,
                    item.absolute_coverage_error_max,
                    3,
                ),
                "width": fmt_seed_range(
                    item.mean_interval_width_bpm_median,
                    item.mean_interval_width_bpm_min,
                    item.mean_interval_width_bpm_max,
                    2,
                ),
                "seeds": fmt_int(item.seed_count),
            }
        )
    return markdown_table(
        rows,
        [
            ("regime", "Evaluation regime"),
            ("mode", "Mode"),
            ("horizon", "Horizon"),
            ("method", "Calibration analysis"),
            ("adjustment", "Adjustment, median [seed range], bpm"),
            ("picp", "90% PICP, median [seed range]"),
            ("error", "Absolute coverage error, median [seed range]"),
            ("width", "Width, median [seed range], bpm"),
            ("seeds", "Seeds"),
        ],
    )


def build_multiseed_calibration_difference() -> str:
    """Report fixed-five-seed paired user-bootstrap effects at 90% coverage."""
    frame = read_csv("multiseed_balanced_calibration_differences_v0_24_0.csv")
    frame = frame[
        (frame["scope"] == "user_bootstrap_after_averaging_five_fixed_seeds")
        & np.isclose(frame["nominal_coverage"], 0.90)
    ].copy()
    rows = []
    for _, item in frame.sort_values(
        ["regime", "mode", "horizon_seconds"]
    ).iterrows():
        rows.append(
            {
                "regime": REGIME_LABELS.get(item.regime, item.regime),
                "mode": MODE_LABELS.get(item["mode"], item["mode"]),
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "picp": (
                    f"{float(item.delta_picp):.3f} "
                    f"[{float(item.delta_picp_ci_low):.3f}, "
                    f"{float(item.delta_picp_ci_high):.3f}]"
                ),
                "error": (
                    f"{float(item.delta_absolute_coverage_error):.3f} "
                    f"[{float(item.delta_absolute_coverage_error_ci_low):.3f}, "
                    f"{float(item.delta_absolute_coverage_error_ci_high):.3f}]"
                ),
                "width": (
                    f"{float(item.delta_mean_interval_width_bpm):.2f} "
                    f"[{float(item.delta_mean_interval_width_ci_low_bpm):.2f}, "
                    f"{float(item.delta_mean_interval_width_ci_high_bpm):.2f}]"
                ),
                "users": fmt_int(item.evaluation_users),
                "replicates": fmt_int(item.bootstrap_replicates),
            }
        )
    return markdown_table(
        rows,
        [
            ("regime", "Evaluation regime"),
            ("mode", "Inference mode"),
            ("horizon", "Horizon"),
            ("picp", "Delta PICP [95% CI]"),
            ("error", "Delta absolute coverage error [95% CI]"),
            ("width", "Delta width [95% CI], bpm"),
            ("users", "Users"),
            ("replicates", "Bootstrap replicates"),
        ],
    )


def build_clustered_calibration() -> str:
    """Summarize calibration-user bootstrap sensitivity for 90% intervals."""
    frame = read_csv("clustered_calibration_bootstrap_v0_20_0.csv")
    frame = frame[np.isclose(frame["nominal_coverage"], 0.90)].copy()
    rows = []
    for _, item in frame.iterrows():
        rows.append(
            {
                "regime": REGIME_LABELS.get(item.regime, item.regime),
                "mode": MODE_LABELS.get(item["mode"], item["mode"]),
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "adjustment": (
                    f"{float(item.adjustment_median_bpm):.3f} "
                    f"[{float(item.adjustment_ci_low_bpm):.3f}, "
                    f"{float(item.adjustment_ci_high_bpm):.3f}]"
                ),
                "picp": (
                    f"{float(item.picp_median):.3f} "
                    f"[{float(item.picp_ci_low):.3f}, {float(item.picp_ci_high):.3f}]"
                ),
                "cal_users": fmt_int(item.calibration_users),
                "eval_users": fmt_int(item.evaluation_users),
            }
        )
    return markdown_table(
        rows,
        [
            ("regime", "Evaluation regime"),
            ("mode", "Mode"),
            ("horizon", "Horizon"),
            ("adjustment", "Median adjustment [95% interval], bpm"),
            ("picp", "Median evaluation PICP [95% interval]"),
            ("cal_users", "Calibration users"),
            ("eval_users", "Evaluation users"),
        ],
    )


def build_external_sport_standardization() -> str:
    """Report natural, sport-matched, and standardized external-minus-internal gaps."""
    frame = read_csv("external_sport_standardization_v0_20_1.csv")
    scope_labels = {
        "reported_natural_mix": "Reported natural mix",
        "shared_family_natural_mix": "Shared-three-family natural mix",
        "sport_matched": "Sport matched",
        "equal_family_standardized": "Equal-family standardized",
        "endomondo_session_mix_standardized": "Standardized to Endomondo session mix",
        "goldencheetah_session_mix_standardized": "Standardized to GoldenCheetah session mix",
    }
    rows = []
    for (scope, sport), group in frame.groupby(
        ["comparison_scope", "sport_family"], sort=False
    ):
        row = {
            "scope": scope_labels.get(scope, scope),
            "sport": str(sport).replace("_", " "),
        }
        for _, item in group.sort_values("horizon_seconds").iterrows():
            horizon = HORIZON_LABELS[int(item.horizon_seconds)]
            row[horizon] = (
                f"{float(item.external_minus_internal_bpm):.3f} "
                f"[{float(item.difference_ci_low_bpm):.3f}, "
                f"{float(item.difference_ci_high_bpm):.3f}]"
            )
        rows.append(row)
    return markdown_table(
        rows,
        [
            ("scope", "Comparison/standardization"),
            ("sport", "Sport family or target mix"),
            ("1 min", "1-min external minus internal MAE [95% CI]"),
            ("3 min", "3-min external minus internal MAE [95% CI]"),
            ("5 min", "5-min external minus internal MAE [95% CI]"),
        ],
    )


def build_external_sport_uncertainty() -> str:
    """Report sport-specific empirical interval behaviour in frozen external data."""
    frame = read_csv("external_sport_uncertainty_bootstrap_v0_21_0.csv")
    rows = []
    for _, item in frame.iterrows():
        rows.append(
            {
                "sport": str(item.sport_family).replace("_", " "),
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "picp": (
                    f"{float(item.picp_90):.3f} "
                    f"[{float(item.picp_90_ci_low):.3f}, {float(item.picp_90_ci_high):.3f}]"
                ),
                "width": (
                    f"{float(item.mean_90_interval_width_bpm):.2f} "
                    f"[{float(item.mean_90_interval_width_bpm_ci_low):.2f}, "
                    f"{float(item.mean_90_interval_width_bpm_ci_high):.2f}]"
                ),
                "wis": (
                    f"{float(item.weighted_interval_score):.2f} "
                    f"[{float(item.weighted_interval_score_ci_low):.2f}, "
                    f"{float(item.weighted_interval_score_ci_high):.2f}]"
                ),
                "users": fmt_int(item.users),
            }
        )
    return markdown_table(
        rows,
        [
            ("sport", "External sport family"),
            ("horizon", "Horizon"),
            ("picp", "90% PICP [95% CI]"),
            ("width", "90% width [95% CI], bpm"),
            ("wis", "WIS [95% CI]"),
            ("users", "Users"),
        ],
    )


def build_external_uncertainty_standardization() -> str:
    """Report matched and sport-standardized point estimates for interval metrics."""
    frame = read_csv("external_sport_uncertainty_standardization_v0_24_0.csv")
    scope_labels = {
        "shared_family_natural_mix": "Shared-three-family natural mix",
        "sport_matched": "Sport matched",
        "equal_family_standardized": "Equal-family standardized",
        "endomondo_user_mix_standardized": "Standardized to Endomondo user mix",
        "goldencheetah_user_mix_standardized": "Standardized to GoldenCheetah user mix",
        "endomondo_session_mix_standardized": "Standardized to Endomondo session mix",
        "goldencheetah_session_mix_standardized": "Standardized to GoldenCheetah session mix",
    }
    metric_labels = {
        "picp_90": "90% PICP",
        "mean_90_interval_width_bpm": "90% interval width",
        "weighted_interval_score": "WIS (lower is better)",
    }
    rows = []
    for (scope, sport, metric), group in frame.groupby(
        ["comparison_scope", "sport_family", "metric"],
        sort=False,
    ):
        row = {
            "scope": scope_labels.get(scope, str(scope).replace("_", " ")),
            "sport": str(sport).replace("_", " "),
            "metric": metric_labels.get(metric, str(metric).replace("_", " ")),
        }
        digits = 3 if metric != "mean_90_interval_width_bpm" else 2
        for _, item in group.sort_values("horizon_seconds").iterrows():
            horizon = HORIZON_LABELS[int(item.horizon_seconds)]
            row[horizon] = (
                f"I {float(item.internal_estimate):.{digits}f}; "
                f"E {float(item.external_estimate):.{digits}f}; "
                f"delta {float(item.external_minus_internal):.{digits}f} "
                f"[{float(item.difference_ci_low):.{digits}f}, "
                f"{float(item.difference_ci_high):.{digits}f}]"
            )
        rows.append(row)
    return markdown_table(
        rows,
        [
            ("scope", "Comparison/standardization"),
            ("sport", "Sport family or target mix"),
            ("metric", "Metric"),
            ("1 min", "1 min: internal; external; delta [95% CI]"),
            ("3 min", "3 min: internal; external; delta [95% CI]"),
            ("5 min", "5 min: internal; external; delta [95% CI]"),
        ],
    )


def build_source_shift_characterization() -> str:
    """Summarize user-balanced descriptive differences between evaluation sources."""
    frame = read_csv("source_shift_characterization_v0_21_0.csv")
    rows = []
    for _, item in frame.iterrows():
        if pd.isna(item.difference_ci_low) or pd.isna(item.difference_ci_high):
            difference = (
                f"{float(item.external_minus_internal):.3f} "
                "[not estimable: both sources constant]"
            )
        else:
            difference = (
                f"{float(item.external_minus_internal):.3f} "
                f"[{float(item.difference_ci_low):.3f}, "
                f"{float(item.difference_ci_high):.3f}]"
            )
        rows.append(
            {
                "category": str(item.category).replace("_", " "),
                "metric": str(item.metric).replace("_", " "),
                "internal": fmt_num(item.internal_point, 3),
                "external": fmt_num(item.external_point, 3),
                "difference": difference,
                "unit": item.unit,
            }
        )
    return markdown_table(
        rows,
        [
            ("category", "Category"),
            ("metric", "Metric"),
            ("internal", "Internal"),
            ("external", "External"),
            ("difference", "External minus internal [95% CI]"),
            ("unit", "Unit"),
        ],
    )


def build_cross_source_duplicate_audit() -> str:
    """Create a compact stage-wise summary from the cross-source audit record."""
    audit = read_json("cross_source_signal_duplicate_audit_v0_20_0.json")
    exact = audit["exact_normalized_signal_audit"]
    hr = audit["exact_hr_subset_screen"]
    near = audit["near_duplicate_candidate_screen"]
    lsh = near["random_hyperplane_lsh"]
    status = near["verification_status_counts"]
    rows = [
        {
            "stage": "Processed modelling sessions",
            "pairs": "Not applicable",
            "result": (
                f"{audit['scope']['source_session_counts']['Endomondo']:,} Endomondo; "
                f"{audit['scope']['source_session_counts']['GoldenCheetah']:,} GoldenCheetah"
            ),
        },
        {
            "stage": "Exact full normalized-signal fingerprint",
            "pairs": fmt_int(exact["cross_source_pairs"]),
            "result": "No confirmed cross-source exact match",
        },
        {
            "stage": "Exact HR-only fingerprint screen",
            "pairs": fmt_int(hr["cross_source_pairs"]),
            "result": "No HR-only exact candidate",
        },
        {
            "stage": "Quantized deterministic profile screen",
            "pairs": fmt_int(near["quantized_signature_candidate_pairs_after_duration_filter"]),
            "result": "No signature candidate",
        },
        {
            "stage": "Random-hyperplane LSH after duration filter",
            "pairs": fmt_int(lsh["unique_pairs_after_duration_filter"]),
            "result": f"{int(lsh['pairs_passing_continuous_prefilter']):,} passed the HR prefilter",
        },
        {
            "stage": "Continuous verification",
            "pairs": fmt_int(status.get("screened_out_after_continuous_verification", 0)),
            "result": "Zero verified HR or HR-plus-auxiliary near-duplicate pairs",
        },
    ]
    return markdown_table(
        rows,
        [("stage", "Audit stage"), ("pairs", "Cross-source pairs"), ("result", "Result")],
    )


def build_persistence_conformal_baseline() -> str:
    frame = read_csv("persistence_conformal_baseline_v0_26_0.csv")
    regime_labels = {
        "strict_temporal_test": "Strict temporal",
        "unseen_user_test": "Unseen user",
        "goldencheetah_frozen_external": "Frozen cross-source",
    }
    rows = []
    for _, item in frame.iterrows():
        rows.append(
            {
                "regime": regime_labels[str(item.regime)],
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "mae": fmt_num(item.mae_bpm, 3),
                "radius": " / ".join(
                    fmt_num(item[f"radius_{level}_bpm"], 2)
                    for level in (50, 80, 90)
                ),
                "picp": " / ".join(
                    fmt_num(item[f"picp_{level}"], 3) for level in (50, 80, 90)
                ),
                "width": " / ".join(
                    fmt_num(item[f"width_{level}_bpm"], 2)
                    for level in (50, 80, 90)
                ),
                "wis": fmt_num(item.weighted_interval_score, 3),
                "support": (
                    f"{fmt_int(item.users)} / {fmt_int(item.sessions)} / "
                    f"{fmt_int(item.origins)}"
                ),
            }
        )
    return markdown_table(
        rows,
        [
            ("regime", "Regime"),
            ("horizon", "Horizon"),
            ("mae", "MAE, bpm"),
            ("radius", "Calibration radius 50/80/90%, bpm"),
            ("picp", "PICP 50/80/90%"),
            ("width", "Width 50/80/90%, bpm"),
            ("wis", "WIS"),
            ("support", "Users / sessions / origins"),
        ],
    )


def build_matched_sport_availability() -> str:
    frame = read_csv("matched_sport_availability_v0_27_0.csv")
    family_labels = {
        "outdoor_cycling": "Outdoor cycling",
        "indoor_virtual_cycling": "Indoor/virtual cycling",
        "running": "Running",
        "walking_hiking": "Walking/hiking",
        "strength_cross_training": "Strength/cross-training",
    }
    rows = []
    for _, item in frame.iterrows():
        rows.append(
            {
                "family": family_labels[str(item.held_sport_family)],
                "horizon": HORIZON_LABELS[int(item.horizon_seconds)],
                "mae": (
                    f"{fmt_num(item.full_sport_mae_bpm, 3)} / "
                    f"{fmt_num(item.held_sport_mae_bpm, 3)}"
                ),
                "delta": (
                    f"{fmt_num(item.held_minus_full_delta_mae_bpm, 3)} "
                    f"[{fmt_num(item.ci_low_bpm, 3)}, {fmt_num(item.ci_high_bpm, 3)}]"
                ),
                "users_higher": f"{float(item.users_with_higher_held_error_percent):.1f}%",
                "support": (
                    f"{fmt_int(item.users)} / {fmt_int(item.sessions)} / "
                    f"{fmt_int(item.origins)}"
                ),
            }
        )
    return markdown_table(
        rows,
        [
            ("family", "Held sport family"),
            ("horizon", "Horizon"),
            ("mae", "Full-sport / held-sport MAE, bpm"),
            ("delta", "Held minus full MAE [95% CI], bpm"),
            ("users_higher", "Users with higher held error"),
            ("support", "Users / sessions / origins"),
        ],
    )


def build_deliberately_leaky_point_control() -> str:
    summary = pd.read_csv(
        LEAKY_AGGREGATION / "paired_metrics_seed_summary_v0_28_0.csv"
    ).set_index("horizon_seconds")
    bootstrap = pd.read_csv(
        LEAKY_AGGREGATION / "paired_user_bootstrap_v0_28_0.csv"
    )
    bootstrap = bootstrap[
        (bootstrap.metric_family == "point") & (bootstrap.metric == "mae_bpm")
    ].set_index("horizon_seconds")
    rows = []
    for horizon in (60, 180, 300):
        item = summary.loc[horizon]
        effect = bootstrap.loc[horizon]
        rows.append(
            {
                "horizon": HORIZON_LABELS[horizon],
                "clean": fmt_seed_range(
                    item.clean_mae_bpm_median,
                    item.clean_mae_bpm_minimum,
                    item.clean_mae_bpm_maximum,
                ),
                "leaky": fmt_seed_range(
                    item.leaky_mae_bpm_median,
                    item.leaky_mae_bpm_minimum,
                    item.leaky_mae_bpm_maximum,
                ),
                "effect": (
                    f"{fmt_num(effect.leaky_minus_clean_estimate, 3)} "
                    f"[{fmt_num(effect.ci_low, 3)}, {fmt_num(effect.ci_high, 3)}]"
                ),
                "relative": fmt_seed_range(
                    item.relative_mae_optimism_percent_median,
                    item.relative_mae_optimism_percent_minimum,
                    item.relative_mae_optimism_percent_maximum,
                ),
                "support": f"{fmt_int(effect.users)} / {fmt_int(effect.matched_seed_count)}",
            }
        )
    return markdown_table(
        rows,
        [
            ("horizon", "Horizon"),
            ("clean", "Clean MAE median [seed range], bpm"),
            ("leaky", "Contaminated MAE median [seed range], bpm"),
            ("effect", "Contaminated minus clean MAE [95% CI], bpm"),
            ("relative", "Apparent MAE optimism median [seed range], %"),
            ("support", "Users / matched seeds"),
        ],
    )


def build_deliberately_leaky_interval_control() -> str:
    diagnostics = pd.read_csv(
        LEAKY_AGGREGATION / "interval_diagnostics_per_seed_v0_28_0.csv"
    )
    diagnostics = diagnostics[
        (diagnostics.nominal_coverage == 0.9) & (diagnostics.calibrated == True)
    ]
    bootstrap = pd.read_csv(
        LEAKY_AGGREGATION / "paired_user_bootstrap_v0_28_0.csv"
    )
    bootstrap = bootstrap[
        (bootstrap.metric_family == "interval")
        & (bootstrap.nominal_coverage == 0.9)
        & (bootstrap.calibrated == True)
    ]
    rows = []
    for horizon in (60, 180, 300):
        seed_rows = diagnostics[diagnostics.horizon_seconds == horizon]
        picp = bootstrap[
            (bootstrap.horizon_seconds == horizon) & (bootstrap.metric == "picp")
        ].iloc[0]
        width = bootstrap[
            (bootstrap.horizon_seconds == horizon)
            & (bootstrap.metric == "mean_interval_width_bpm")
        ].iloc[0]
        rows.append(
            {
                "horizon": HORIZON_LABELS[horizon],
                "picp": (
                    f"{float(seed_rows.clean_picp.median()):.3f} / "
                    f"{float(seed_rows.leaky_picp.median()):.3f}"
                ),
                "delta_picp": (
                    f"{fmt_num(picp.leaky_minus_clean_estimate, 3)} "
                    f"[{fmt_num(picp.ci_low, 3)}, {fmt_num(picp.ci_high, 3)}]"
                ),
                "width": (
                    f"{float(seed_rows.clean_mean_interval_width_bpm.median()):.2f} / "
                    f"{float(seed_rows.leaky_mean_interval_width_bpm.median()):.2f}"
                ),
                "delta_width": (
                    f"{fmt_num(width.leaky_minus_clean_estimate, 3)} "
                    f"[{fmt_num(width.ci_low, 3)}, {fmt_num(width.ci_high, 3)}]"
                ),
            }
        )
    return markdown_table(
        rows,
        [
            ("horizon", "Horizon"),
            ("picp", "Clean / contaminated 90% PICP median"),
            ("delta_picp", "PICP difference [95% CI]"),
            ("width", "Clean / contaminated width median, bpm"),
            ("delta_width", "Width difference [95% CI], bpm"),
        ],
    )


def build_horizon_specific_eligibility() -> str:
    frame = read_csv("horizon_specific_eligibility_v0_29_0.csv")
    rows = []
    for regime in (
        "within_user_temporal_test",
        "unseen_user_test",
        "goldencheetah_frozen_external",
    ):
        for horizon in (60, 180, 300):
            selected = frame[
                (frame.regime == regime) & (frame.horizon_seconds == horizon)
            ].set_index("cohort")
            common = selected.loc["common_three_target"]
            specific = selected.loc["horizon_specific"]
            added_percent = 100.0 * float(specific.added_origins_vs_common) / float(
                common.origins
            )
            rows.append(
                {
                    "regime": REGIME_LABELS[regime],
                    "horizon": HORIZON_LABELS[horizon],
                    "common_origins": fmt_int(common.origins),
                    "specific_origins": fmt_int(specific.origins),
                    "added": (
                        f"{fmt_int(specific.added_origins_vs_common)} "
                        f"({added_percent:.1f}%)"
                    ),
                    "common_mae": fmt_num(common.hierarchical_mae_bpm, 3),
                    "specific_mae": fmt_num(specific.hierarchical_mae_bpm, 3),
                    "delta": (
                        f"{fmt_num(specific.mae_delta_vs_common_bpm, 3)} "
                        f"[{fmt_num(specific.delta_ci_lower_bpm, 3)}, "
                        f"{fmt_num(specific.delta_ci_upper_bpm, 3)}]"
                    ),
                }
            )
    return markdown_table(
        rows,
        [
            ("regime", "Regime"),
            ("horizon", "Horizon"),
            ("common_origins", "Common-cohort origins"),
            ("specific_origins", "Horizon-specific origins"),
            ("added", "Added origins (%)"),
            ("common_mae", "Common-cohort MAE, bpm"),
            ("specific_mae", "Horizon-specific MAE, bpm"),
            ("delta", "MAE difference [95% CI], bpm"),
        ],
    )


def build_horizon_specific_frozen_models() -> str:
    frame = read_csv("horizon_specific_frozen_model_summary_v0_30_0.csv")
    rows = []
    for regime in (
        "within_user_temporal_test",
        "unseen_user_test",
        "goldencheetah_frozen_external",
    ):
        for horizon in (60, 180, 300):
            item = frame[
                (frame.regime == regime) & (frame.horizon_seconds == horizon)
            ].iloc[0]
            rows.append(
                {
                    "regime": (
                        f"{REGIME_LABELS[regime]} "
                        f"({MODE_LABELS[str(item['mode'])].lower()})"
                    ),
                    "horizon": HORIZON_LABELS[horizon],
                    "common_mae": fmt_seed_range(
                        item.common_mae_median_bpm,
                        item.common_mae_min_bpm,
                        item.common_mae_max_bpm,
                    ),
                    "specific_mae": fmt_seed_range(
                        item.horizon_specific_mae_median_bpm,
                        item.horizon_specific_mae_min_bpm,
                        item.horizon_specific_mae_max_bpm,
                    ),
                    "seed_delta": fmt_seed_range(
                        item.expanded_minus_common_seed_median_bpm,
                        item.expanded_minus_common_seed_min_bpm,
                        item.expanded_minus_common_seed_max_bpm,
                    ),
                    "paired_delta": (
                        f"{fmt_num(item.paired_user_estimate_bpm)} "
                        f"[{fmt_num(item.paired_user_ci_lower_bpm)}, "
                        f"{fmt_num(item.paired_user_ci_upper_bpm)}]"
                    ),
                    "added": fmt_int(item.added_origins),
                }
            )
    return markdown_table(
        rows,
        [
            ("regime", "Regime and inference mode"),
            ("horizon", "Horizon"),
            ("common_mae", "Common MAE median [seed range], bpm"),
            ("specific_mae", "Expanded MAE median [seed range], bpm"),
            ("seed_delta", "Expanded - common median [seed range], bpm"),
            ("paired_delta", "Paired-user difference [95% CI], bpm"),
            ("added", "Added origins"),
        ],
    )


def main():
    flow, support = build_dataset_flow()
    text = f"""# Supplementary material

## Uncertainty-Aware Exercise Heart-Rate Forecasting under User and Sport Distribution Shifts: A Leakage-Controlled Multi-Dataset Study

**Target journal:** *Biomedical Signal Processing and Control*

## Supplementary methods

All supplementary results use the same session inclusion, exact-signal duplicate control, sport ontology, past-only 5-min context, 1/3/5-min targets, split-before-windowing rule, completed-session history rule, training-only normalization, checkpoint selection, and session-then-user hierarchical aggregation described in the main manuscript. The primary analyses required all three targets, so every horizon uses a common complete-three-target cohort. Table S19 isolates that eligibility decision with a fixed-session, parameter-free persistence diagnostic and a five-seed frozen-main-model sensitivity. The principal history-capable and zero-history-trained TCN models were repeated with five seeds, whereas GRU, point-TCN, and held-sport experiments used three seeds; deterministic baselines have no optimization seed, and explicitly labelled secondary sensitivities use the frozen reference seed 20260722. Seed summaries are reported as medians [minimum--maximum] and are descriptive stability ranges, not confidence intervals.

GoldenCheetah athlete-directory identifiers defined users. Within each directory, the local timestamp encoded by a CSV filename was linked to JSON ride metadata only when exactly one match arose after testing UTC offsets from -14:00 to +14:00 in 15-min steps. Of 51,470 CSVs in 150 directories, 50,002 (97.15%) were linked uniquely; 1,335 records had missing or invalid metadata, 111 had ambiguous matches, 14 were unmatched, and 8 had duplicate metadata matches. Signal-quality and duplicate controls then yielded 32,587 modelling sessions from 144 users. This outcome-blind linkage used no HR target values.

Raw observations were assigned to right-closed 10-s bins and the last valid HR, speed, or altitude value in each bin was retained; empty value fields remained zero with a separate observed/missing mask. The 13 completed-history variables were log prior-session count; log mean and log standard deviation of session duration; mean and standard deviation of prior session-mean HR; mean prior within-session HR standard deviation; mean and standard deviation of prior session-mean speed; mean prior within-session altitude standard deviation; log same-sport session count; same-sport mean HR; same-sport mean speed; and log days since the preceding session start. `log1p` was used for counts, duration quantities, and recency. For unseen test users, earlier workouts in the chronological evaluation stream could update these summaries only after ending; no model parameter was updated. In held-family experiments, the held family was absent from training, validation, calibration, and history updates; its sport token was remapped to learned other/unknown code 0, and 0.1 sport-token dropout exposed that code during training on nonheld families. GoldenCheetah uses history-masked inference from the history-capable model, so it evaluates cross-source transport without prior-workout input rather than transport of the completed-history branch. “Unseen user” means absent from fitting and is not synonymous with cold start or early onboarding.

The main network projected six value/mask channels to 64 channels and used four residual TCN blocks, each with two kernel-3 causal convolutions, channel layer normalization, GELU activation, and 0.1 dropout; block dilations were 1, 2, 4, and 8. The history encoder was a 13--32--32 GELU MLP with a learned 32-dimensional no-history vector. An eight-dimensional sport embedding, log elapsed time, the 64-dimensional current state, the history vector, and its presence mask entered a 96-unit GELU fusion layer with 0.1 dropout and 21 outputs. The mean pinball loss weighted all seven quantiles and three horizons equally; outputs were sorted by quantile. Formal main runs used AdamW with learning rate 0.001, weight decay 0.0001, mixed precision, gradient-norm clipping at 1.0, a plateau scheduler (factor 0.5, patience 1, minimum learning rate 0.00001), batch size 2,048, 500,000 sampled origins per epoch, at most 40 epochs, and early-stopping patience 4. Held-family runs used 250,000 samples per epoch, patience 3, and 0.1 sport-token dropout. GRU used two 64-unit layers; point-TCN used the same 64-channel four-block encoder; Transformer used three 64-dimensional, four-head layers with 128-unit feed-forward sublayers. XGBoost used up to 2,000 depth-8 histogram trees, learning rate 0.03, 0.85 row and column subsampling, and 100-round early stopping.

For the zero-history-trained strategy contrasts, paired per-user differences were first averaged over the five matched seeds and then bootstrapped over users with 10,000 replicates. These intervals therefore quantify user-sampling variation conditional on the declared seed set; seeds were not resampled and full optimization uncertainty is not covered. Reference-seed confidence intervals use 10,000 user-clustered bootstrap replicates conditional on that checkpoint. The mean-effect confidence interval is primary comparison evidence; Holm-adjusted paired Wilcoxon p-values are complementary rank evidence and do not override confidence intervals that include zero. CQR thresholds were estimated by pooling correlated calibration origins and therefore do not provide a finite-sample guarantee for equal-user PICP; interval tables report empirical post-CQR performance.

The authoritative multiseed artifacts are the v0.22 aggregation under `outputs/q1_multiseed_v0_21_0/aggregation` and the v0.23 zero-history-trained aggregation under `outputs/independent_zero_history_v0_23_0/aggregation`. Frozen-prediction v0.24 analyses add five-seed equal-user/equal-session calibration and reference-seed sport-composition standardization of interval metrics; v0.25 provides paired-user comparisons; v0.26 provides an independently calibrated persistence interval baseline without model fitting; and v0.27 provides a post hoc matched-origin sport-availability sensitivity from frozen predictions. Version 0.28 is a separately trained, deliberately invalid negative control in which non-evaluation origins from strict-temporal test sessions contaminate fitting, validation, and calibration; it is excluded from valid-model rankings. Version 0.29 provides a parameter-free, fixed-session target-availability sensitivity reconstructed from the raw HR streams. Version 0.30 applies the five frozen main-model seeds to only the additional horizon-eligible rows, with no training, checkpoint selection, normalization refit, calibration, or external adaptation. Fifteen full-batch inference replays and 45 common-cohort hierarchical-MAE checks reproduced the authoritative values exactly. The v0.24--v0.27 post-processing analyses did not adapt or recalibrate on GoldenCheetah, and v0.29 did not fit, adapt, or calibrate a model. The final reference-seed user/cross-source paired-comparison artifact is `outputs/results/paired_model_comparisons_v0_11_0.csv`; the earlier `paired_user_bootstrap_v0_11_0.csv` is retained for provenance but is superseded because it preceded the final aligned prediction rebuild. All values below are generated directly from the authoritative result files by `src/build_supplementary_material.py`.

## Table S1. Dataset construction and evaluation support

### Table S1a. Session and forecast-origin flow

{flow}

The session-eligible split contained 1,090 Endomondo users, whereas 1,085 users contributed at least one accepted forecast origin. Table S1a reports users represented after origin acceptance; the unseen-user assignments in Table S1b were made earlier, before origin construction.

### Table S1b. Principal partition support

{support}

## Table S2. Full strict-temporal point-forecast metrics

MAE is in beats per minute (bpm). Learned-model entries are medians [seed range], with the number of seeds shown explicitly; deterministic baselines are single fixed evaluations.

{build_temporal_point()}

## Table S3. Full unseen-user and frozen cross-source point-forecast metrics

All GoldenCheetah rows use frozen Endomondo preprocessing and history-masked checkpoints. No target-source adaptation or recalibration was performed, and these rows do not test transfer of the completed-history branch. Learned-model entries are medians [seed range], with the number of seeds shown explicitly; single-run secondary comparators are labelled.

{build_user_external_point()}

## Table S4. Raw and conformalized interval performance

### Table S4a. Raw and post-CQR point estimates

PICP is prediction-interval coverage probability; width and conformal adjustment are in bpm. CQR denotes conformalized quantile regression. Entries are medians [seed range] across five seeds; seed ranges are not confidence intervals. Strict-temporal and unseen-user rows use history-informed inference, whereas GoldenCheetah rows use history-masked inference and apply Endomondo-derived adjustments unchanged. Figure 4b instead uses the explicitly matched unseen-user history-masked mode.

{build_uncertainty()}

### Table S4b. User-bootstrap uncertainty for primary regime modes

All rows use post-CQR 90% intervals from the frozen reference seed 20260722. The Mode column makes the information state explicit: unseen-user rows here are history-informed, while the matched comparison in Figure 4b is history-masked. Brackets are 95% confidence intervals from 10,000 user bootstrap replicates. Width-error Spearman is first calculated within each eligible user and then averaged across users. The intervals quantify user-sampling variation conditional on that fitted checkpoint and should not be interpreted as multiseed intervals.

{build_figure3_uncertainty()}

## Table S5. Leave-one-sport-family-out and joint-shift results

### Table S5a. Point performance and support

History-informed hierarchical MAE is reported as the median [seed range] across three seeds. Seed ranges are descriptive and are not confidence intervals. Joint intersections below 25 users are explicitly cautionary.

{build_sport_shift()}

### Table S5b. Empirical post-CQR 90% interval performance

PICP and interval width are aggregated within session and then user and reported as medians [seed range] across three seeds. Low-support joint intersections remain cautionary.

{build_sport_uncertainty()}

### Table S5c. Three-seed paired-user main-versus-EWMA effects

Within each seed, MAE differences were aggregated within session and then user; each user's effect was averaged across the three matched seeds before 10,000 user-bootstrap resamples. Negative values favour the history-informed main model. Joint user--sport rows are exploratory regardless of user count, and rows below 25 users receive an additional caution flag.

{build_sport_paired_bootstrap_v025()}

## Table S6. Model and ablation effects

### Table S6a. Seed-paired main-versus-comparator effects

Delta is MAE(left strategy) minus MAE(right strategy); negative values favour the left strategy. History contrasts use matched history-capable checkpoints, whereas comparator contrasts use history-masked inference. Entries are medians [seed range] over matched seeds and are descriptive rather than inferential confidence intervals.

{build_seed_paired_effects()}

### Table S6b. Zero-history-trained strategy contrasts

Delta is MAE(strategy A) minus MAE(strategy B); negative values favour strategy A. Paired per-user differences were averaged over five matched seeds before 10,000 user-bootstrap resamples. The resulting 95% intervals condition on the declared seed set and do not resample seeds.

{build_independent_strategy_effects()}

### Table S6c. Three-seed paired-user main-versus-GRU/TCN effects

The history-capable main model is evaluated with prior-workout input masked to match the established architecture-comparison estimand. Per-user effects were averaged across three matched seeds before 10,000 user-bootstrap resamples. Negative values favour the main model; these post hoc intervals condition on the declared seeds.

{build_model_paired_bootstrap_v025()}

### Table S6d. Reference-seed signal-ablation effects

These paired user-level multimodal-minus-HR-only effects use the frozen reference seed 20260722. Confidence intervals, rather than rank-test p-values alone, govern mean-effect claims.

{build_reference_seed_effects()}

## Table S7. Evaluation-origin stride sensitivity

All predictions use the frozen unseen-user reference checkpoint (seed 20260722). Values are hierarchical MAE in bpm.

{build_stride()}

## Table S8. Recorded-gender descriptive contrasts

These frozen-reference-seed (20260722) results are unadjusted platform-recorded subgroup descriptions, not biological, causal, fairness, or clinical comparisons. Every confidence interval includes zero; the unseen-user recorded-female subgroup contains only 10 users.

{build_gender()}

## Table S9. Targeted evidence comparison with direct HR-modelling studies

The comparison uses the frozen 41-record project bibliography originally reconciled to the author-approved Zotero collection, together with a documented targeted update conducted on 23 July 2026; it is not a systematic review and cannot support a global first/only claim. Later unrelated Zotero collection-membership drift is audited separately and did not alter this bibliography. `NR` means not reported or not established from the inspected primary evidence; it does not mean no. Current-window HR estimation, past-only future forecasting, and forecasting with known future route information are kept distinct. Study names refer to the numbered references in the main manuscript.

{build_prior_work()}

## Table S10. Past-only completed-workout history availability

Counts are evaluated over unique 300-s-reporting test sessions. Prior-session counts include only workouts that ended no later than the current workout start. Q1 and Q3 denote session-level quartiles. A history-informed checkpoint still uses its explicit no-history state when no completed workout is available.

{build_history_availability()}

## Table S11. Calibration-estimand and clustered-calibration sensitivity

### Table S11a. Five-seed origin-pooled versus equal-user/session calibration

This five-seed sensitivity gives each calibration user equal influence by weighting sessions equally within user and origins equally within session. Values are medians [seed ranges]. The resulting threshold is a weighted empirical estimand-matching analysis, not a distribution-free guarantee. GoldenCheetah rows reuse the corresponding Endomondo history-masked thresholds without target-data recalibration. Strict-temporal calibration predictions were not persisted and were not reconstructed from test targets.

{build_user_balanced_calibration()}

### Table S11b. Five-seed paired user-bootstrap calibration differences

Differences are equal-user/equal-session minus origin-pooled results after each user's metrics were averaged across the five fixed seeds. Positive PICP and width differences indicate larger values under equal-user/equal-session calibration; negative absolute-coverage-error differences favour that sensitivity. Confidence intervals use 10,000 paired user resamples, condition on both estimated thresholds and the declared seeds, and do not include calibration-sample or optimization-seed uncertainty.

{build_multiseed_calibration_difference()}

### Table S11c. Calibration-user bootstrap sensitivity at the reference seed

Using the reference seed 20260722, each replicate resamples calibration users and one session and origin within each sampled user before estimating the threshold. Entries are medians and 2.5th--97.5th percentiles over 10,000 replicates. Wide threshold intervals reflect limited calibration-user support and clustered time-series dependence; no formal finite-sample coverage guarantee is claimed.

{build_clustered_calibration()}

## Table S12. Cross-source sport-composition standardization

### Table S12a. Point-error standardization

All comparisons use history-masked predictions from the same frozen reference checkpoint (seed 20260722). Differences are GoldenCheetah minus Endomondo hierarchical MAE in bpm. Sport-matched and standardized estimates are descriptive contrasts; they do not identify causal platform or device effects. The shared-family analyses are restricted to outdoor cycling, indoor/virtual cycling, and running.

{build_external_sport_standardization()}

### Table S12b. Interval-metric standardization under matched history masking

All comparisons use the same frozen reference-seed history-capable checkpoint with prior-workout input masked in Endomondo and GoldenCheetah. I denotes Endomondo, E denotes GoldenCheetah, and delta is E minus I. PICP, 90% width, and WIS are aggregated within session and then user; WIS is better when lower. Confidence intervals independently resample users within each source 10,000 times. The estimates are descriptive and do not identify causal platform or device effects or furnish user-level conformal guarantees.

{build_external_uncertainty_standardization()}

## Table S13. Sport-specific frozen cross-source interval performance

All rows use the frozen reference seed 20260722 and apply Endomondo-derived CQR thresholds unchanged. Brackets are 95% intervals from 10,000 user bootstrap replicates and quantify user-sampling variation conditional on that fitted checkpoint.

{build_external_sport_uncertainty()}

## Table S14. Descriptive source-shift characterization

Internal and cross-source inputs are the unseen-user history-masked Endomondo test set and history-masked GoldenCheetah set, respectively, from reference seed 20260722. Metrics are averaged within session and then equally across users; confidence intervals resample users independently within each source. These contrasts jointly reflect users, sports, devices, sampling, session structure, and platform processing and are not causal source or device effects.

{build_source_shift_characterization()}

## Table S15. Cross-source normalized-signal duplicate audit

The exact stage joins cryptographic fingerprints of complete processed 10-s HR, speed, and altitude values and masks after removing identifiers, source labels, absolute timestamps, and sport labels. The approximate stage combines deterministic quantized profiles with random-hyperplane locality-sensitive hashing and continuous verification. A zero verified count reduces contamination concern but cannot prove absence of every duplicate affected by cropping, drift, long gaps, smoothing, or transformations outside the declared search.

{build_cross_source_duplicate_audit()}

## Table S16. Independent symmetric split-conformal persistence baseline

Persistence used the most recent observed context HR as its deterministic point forecast. For each horizon and nominal level, the absolute-residual radius was the finite-sample higher order statistic from the dedicated strict-temporal or unseen-user calibration partition. GoldenCheetah reused the unseen-user Endomondo radii unchanged. Bounds were clipped to 30--240 bpm, and all metrics were averaged within session and then user. This independently calibrated baseline has no learned parameters; it is not a second calibration of the quantile TCN. Because calibration pooled correlated origins, the intervals do not furnish a finite-sample guarantee for session-then-user PICP.

{build_persistence_conformal_baseline()}

## Table S17. Post hoc matched-origin sport-availability sensitivity

For each held family and each of the three shared seeds, history-masked predictions from the full-sport unseen-user model and held-family model were aligned by the same global origin index within the joint unseen-user/sport test. Absolute errors were averaged within session and user, then each user's effect was averaged across seeds before 10,000 user-bootstrap resamples. Positive differences indicate higher error when the sport family was unavailable during fitting and represented by code 0. This is an operational sport-availability contrast, not a causal sport effect: it also captures the held-family model's locked token exposure, sport-excluded fitting data, and training budget. Rows with fewer than 25 users remain cautionary.

{build_matched_sport_availability()}

## Table S18. Deliberately contaminated same-session-window negative control

This retrospective negative control kept the 104,144 strict-temporal test origins unchanged but deliberately assigned 290,245, 62,146, and 62,463 other 60-s origins from those test sessions to fitting, validation, and calibration by a model-seed-independent SHA-256 rule. Exact test rows remained disjoint. Nevertheless, 15,839 of 16,012 test sessions entered fitting, 98.9% of test origins had a contaminated fitting origin within 300 s, and 95.0% shared at least one target timestamp with a contaminated fitting origin. Three freshly initialized zero-history-trained models used the formal budget and were compared with the corresponding clean v0.23 predictions on the identical row order. Per-user differences were averaged over the three matched seeds before 10,000 user-bootstrap resamples. This deliberately invalid pipeline is not eligible for a model leaderboard, and its observed effect is specific to this contamination design rather than a general estimate of leakage bias.

### Table S18a. Point-error contrast

Negative contaminated-minus-clean MAE denotes apparent optimism. Relative optimism is 100 * (clean - contaminated) / clean; negative values denote deterioration. Seed ranges are descriptive, whereas the difference confidence interval is a paired user bootstrap conditional on the three seeds.

{build_deliberately_leaky_point_control()}

### Table S18b. Empirical 90% interval contrast

Intervals use each pipeline's own invalidly contaminated or clean calibration partition as applicable. The CQR guarantee is not valid for the contaminated design. Differences are contaminated minus clean after identical session--user aggregation and three-seed user pairing.

{build_deliberately_leaky_interval_control()}

## Table S19. Horizon-specific target-availability sensitivity

The primary complete-three-target rule conditions every horizon on availability of all 1-, 3-, and 5-min targets. Both post hoc diagnostics held the original users and evaluation sessions fixed, rebuilt the same 300-s reporting grid from the raw heart-rate streams, and required only the target for the horizon being evaluated. Positive expanded-minus-common differences indicate higher error on the enlarged cohort.

### Table S19a. Parameter-free persistence diagnostic

Persistence used the most recent observed 10-s context-bin HR. Errors were averaged within session and then user; confidence intervals used 10,000 paired user resamples. All common-cohort user, session, and origin counts were reproduced exactly, and persistence MAE differed from the authoritative artifacts by less than 0.000001 bpm.

{build_horizon_specific_eligibility()}

### Table S19b. Five-seed frozen-main-model sensitivity

The original saved predictions were retained for common-cohort rows, and only the additional horizon-eligible rows were newly inferred. The strict-temporal and unseen-user tests used their frozen history-informed checkpoints; GoldenCheetah used frozen history-masked inference. No model was retrained or selected, normalizers were not refitted, no interval calibration was performed, and GoldenCheetah supplied no adaptation signal. Seed ranges are descriptive. The paired-user effect first averages each user's expanded-minus-common difference across the five matched seeds and then uses 10,000 user-bootstrap resamples; seeds are not resampled.

{build_horizon_specific_frozen_models()}

## Supplementary figure caption

**Supplementary Fig. 1. Ablation, stride sensitivity, and subgroup boundaries.** (a) Reference-seed paired multimodal-minus-HR-only MAE differences. (b) Reference-seed change in MAE when the frozen unseen-user model is evaluated every 60 s rather than every 300 s. (c) Five-seed history-informed-minus-zero-history-trained effects, with paired per-user differences averaged over seeds before user bootstrap. (d) Reference-seed recorded-female-minus-recorded-male descriptive MAE differences; the unseen-user recorded-female subgroup contains only 10 users.

## Supplementary provenance

- Forecast-origin flow: `outputs/audit/forecast_origins_full_v0_3_1.json`.
- Split support: `outputs/audit/split_manifest_v0_2_0_summary.json`.
- Multiseed main/comparator and interval summaries: `outputs/q1_multiseed_v0_21_0/aggregation/seed_variability_summary_v0_22_0.csv`, `outputs/q1_multiseed_v0_21_0/aggregation/main_history_difference_summary_v0_22_0.csv`, and `outputs/q1_multiseed_v0_21_0/aggregation/main_vs_comparator_summary_v0_22_0.csv`.
- Zero-history-trained strategy summaries: `outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrast_seed_summary_v0_23_0.csv`, `strategy_contrast_user_bootstrap_v0_23_0.csv`, and `strategy_contrasts_per_seed_v0_23_0.csv`.
- Strict temporal metrics: `outputs/results/temporal_uncertainty_point_v0_13_0.csv` and `temporal_aligned_baselines_v0_13_0.csv`.
- User/cross-source metrics: `uncertainty_point_metrics_v0_11_0.csv`, neural/XGBoost comparator files, and `naive_baseline_metrics_v0_5_0.csv`.
- Interval metrics: `temporal_uncertainty_interval_v0_13_0.csv`, `uncertainty_interval_metrics_v0_11_0.csv`, and `figure3_uncertainty_bootstrap_v0_18_0.csv`.
- Sport shift: `sport_shift_point_v0_12_0.csv`, `sport_shift_mae_bootstrap_v0_19_0.csv`, and `sport_shift_uncertainty_bootstrap_v0_17_0.csv`.
- Three-seed paired-user comparator and sport effects: `outputs/results/multiseed_paired_model_comparisons_v0_25_0.csv`, `multiseed_paired_sport_shift_v0_25_0.csv`, and `outputs/audit/multiseed_paired_user_bootstrap_v0_25_0.audit.json`.
- Reference-seed paired effects and sensitivities: `temporal_paired_comparisons_v0_13_0.csv`, `paired_model_comparisons_v0_11_0.csv`, and `signal_ablation_paired_v0_14_0.csv`.
- Stride sensitivity and recorded-gender contrasts: version 0.15.0 and 0.16.0 result artifacts.
- Prior-work coding: `references/PRIOR_WORK_COMPARISON.md` and the documented targeted update `references/TARGETED_LITERATURE_UPDATE_2026-07-23.md`; this is not a systematic review.
- Completed-history availability: `outputs/results/history_availability_v0_19_0.csv`.
- Calibration sensitivity: `outputs/results/multiseed_balanced_calibration_summary_v0_24_0.csv`, `multiseed_balanced_calibration_differences_v0_24_0.csv`, `outputs/audit/multiseed_balanced_calibration_v0_24_0.json`, and `clustered_calibration_bootstrap_v0_20_0.csv`.
- Cross-source sport composition and interval heterogeneity: `external_sport_standardization_v0_20_1.csv`, `external_sport_uncertainty_standardization_v0_24_0.csv`, `outputs/audit/external_sport_uncertainty_standardization_v0_24_0.json`, and `external_sport_uncertainty_bootstrap_v0_21_0.csv`.
- Source-shift characterization: `source_shift_characterization_v0_21_0.csv`, `source_shift_sport_composition_v0_21_0.csv`, and `source_shift_session_distributions_v0_21_0.csv`.
- Cross-source duplicate audit: `outputs/audit/cross_source_signal_duplicate_audit_v0_20_0.json` and its two result CSV files.
- Independent probabilistic baseline: `outputs/results/persistence_conformal_baseline_v0_26_0.csv` and `outputs/audit/persistence_conformal_baseline_v0_26_0.json`.
- Matched-origin sport availability: `outputs/results/matched_sport_availability_v0_27_0.csv` and `outputs/audit/matched_sport_availability_v0_27_0.json`.
- Deliberately contaminated same-session-window negative control: `outputs/deliberately_leaky_negative_control_v0_28_0/aggregation/paired_metrics_seed_summary_v0_28_0.csv`, `paired_user_bootstrap_v0_28_0.csv`, `interval_diagnostics_per_seed_v0_28_0.csv`, and `audit.json`; the per-user bootstrap input remains private.
- Horizon-specific target eligibility: `outputs/results/horizon_specific_eligibility_v0_29_0.csv` and `outputs/audit/horizon_specific_eligibility_v0_29_0.json`.
- Frozen-model horizon-specific target eligibility: `outputs/results/horizon_specific_frozen_model_per_seed_v0_30_0.csv`, `horizon_specific_frozen_model_summary_v0_30_0.csv`, and `outputs/audit/horizon_specific_frozen_models_v0_30_0.json`.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
