from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
Q1_AGGREGATION = ROOT / "outputs" / "q1_multiseed_v0_21_0" / "aggregation"
ZERO_AGGREGATION = ROOT / "outputs" / "independent_zero_history_v0_23_0" / "aggregation"
FIGURES = ROOT / "figures"
SOURCE = FIGURES / "source_data"

COLORS = {
    "history": "#0F4D92",
    "zero": "#6F91BF",
    "ewma": "#767676",
    "persistence": "#B8B8B8",
    "external": "#B64342",
    "temporal": "#42949E",
    "unseen": "#0F4D92",
    "joint": "#9A4D8E",
    "gold": "#C69214",
    "black": "#272727",
    "light": "#E8EDF3",
}
HORIZONS = [60, 180, 300]
HORIZON_LABELS = ["1 min", "3 min", "5 min"]


def style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7
    plt.rcParams["lines.linewidth"] = 1.5


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str,
        face: str, edge: str = "none", fontsize: float = 6.5) -> patches.FancyBboxPatch:
    item = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(item)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=3,
    )
    return item


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#606060") -> None:
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            zorder=1,
        )
    )


def figure1() -> None:
    fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.1, 0.9])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_axis_off()

    panel_label(ax_a, "a")
    ax_a.set_xlim(-0.3, 10.4)
    ax_a.set_ylim(-0.2, 2.0)
    # Keep the future-target time line outside the context block.  Drawing the
    # line through the block made it cross the two-line context label after the
    # figure was reduced to journal page width.
    ax_a.plot([5, 10], [0.8, 0.8], color=COLORS["black"], lw=1, zorder=1)
    ax_a.add_patch(
        patches.Rectangle(
            (0, 0.45),
            5,
            0.7,
            facecolor="#DDE8F3",
            edgecolor=COLORS["history"],
            lw=0.8,
            zorder=2,
        )
    )
    ax_a.text(
        2.5,
        0.8,
        "Causal context: previous 5 min\nHR + speed + altitude",
        ha="center",
        va="center",
        fontsize=7,
        linespacing=1.25,
        zorder=3,
    )
    ax_a.axvline(5, ymin=0.18, ymax=0.72, color=COLORS["black"], lw=1)
    ax_a.text(5, 0.25, "forecast origin", ha="center", va="top", fontsize=6.5)
    for x, label in zip([6, 8, 10], HORIZON_LABELS):
        ax_a.plot(x, 0.8, "o", color=COLORS["external"], ms=4)
        ax_a.text(x, 1.2, label, ha="center", va="bottom", fontsize=7, color=COLORS["external"])
    ax_a.text(8, 1.65, "Future heart-rate targets", ha="center", fontsize=7.5, fontweight="bold")
    ax_a.text(0, 1.65, "Forecasting task", ha="left", fontsize=7.5, fontweight="bold")

    panel_label(ax_b, "b")
    # Keep a visible safety margin around the right-most output node after the
    # figure is reduced and embedded in Word/PDF.  The earlier 0.3-unit margin
    # made the node appear clipped in some cropped/zoomed views.
    ax_b.set_xlim(0, 10.4)
    ax_b.set_ylim(0, 6)
    box(ax_b, (0.2, 3.8), 2.65, 1.15, "Current workout\ncausal TCN", "#DDE8F3", fontsize=6.3)
    box(
        ax_b,
        (0.2, 1.05),
        2.65,
        1.4,
        "Completed prior\nworkouts\n13-feature history",
        "#E3F0ED",
        fontsize=5.9,
    )
    box(
        ax_b,
        (3.65, 2.42),
        2.35,
        1.35,
        "Fusion\n+ sport\n+ elapsed time",
        "#EEE8F3",
        fontsize=6.0,
    )
    box(
        ax_b,
        (6.85, 2.42),
        2.85,
        1.35,
        "7 ordered\nquantiles\nfor 1/3/5 min",
        "#F4E3E1",
        fontsize=6.0,
    )
    arrow(ax_b, (2.85, 4.38), (3.65, 3.35))
    arrow(ax_b, (2.85, 1.75), (3.65, 2.83))
    arrow(ax_b, (6.0, 3.10), (6.85, 3.10))
    ax_b.text(0.2, 5.45, "Uncertainty-aware personalization", fontsize=7.5, fontweight="bold")
    ax_b.text(0.2, 0.45, "History rule: a prior session enters only after it has ended", fontsize=6.3, color=COLORS["black"])

    panel_label(ax_c, "c")
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 8)
    ax_c.text(0.1, 7.45, "Leakage-controlled evaluation", fontsize=7.5, fontweight="bold")
    rows = [
        ("Temporal", ["Train", "Val", "Cal", "Test"], COLORS["temporal"]),
        ("Unseen user", ["Train\nusers", "Val", "Cal", "Test\nusers"], COLORS["unseen"]),
        ("Held sport", ["Seen\nsports", "Val", "Cal", "Held\nsport"], COLORS["gold"]),
        ("Joint shift", ["Seen users\n+ sports", "Val", "Cal", "New user\n+ held sport"], COLORS["joint"]),
        ("Cross-source", ["Endomondo\ntrain", "Val", "Cal", "GoldenCheetah\nhistory-masked"], COLORS["external"]),
    ]
    y_positions = np.linspace(6.3, 1.0, len(rows))
    for (name, labels, color), y in zip(rows, y_positions):
        ax_c.text(0.1, y + 0.35, name, ha="left", va="center", fontsize=6.3, fontweight="bold")
        widths = [2.15, 1.05, 1.05, 2.2]
        x = 2.5
        for label, width, alpha in zip(labels, widths, [0.28, 0.42, 0.55, 0.9]):
            box(ax_c, (x, y), width, 0.7, label, mpl.colors.to_rgba(color, alpha), edge=color, fontsize=4.8)
            x += width + 0.18
    ax_c.text(2.5, 0.15, "GoldenCheetah: three shared families; no adaptation or recalibration", fontsize=5.5)
    save(fig, "Figure_1_study_design")


def line_panel(
    ax: plt.Axes,
    data: dict[str, dict[str, list[float]]],
    title: str,
    label: str,
) -> None:
    mapping = {
        "History-informed": (COLORS["history"], "o", "-"),
        "History-masked": (COLORS["zero"], "s", "-"),
        "Zero-history-trained": (COLORS["joint"], "P", "-"),
        "EWMA": (COLORS["ewma"], "^", "--"),
        "Persistence": (COLORS["persistence"], "D", ":"),
        "GRU": (COLORS["black"], "v", "--"),
    }
    for name, summary in data.items():
        color, marker, linestyle = mapping[name]
        values = np.asarray(summary["median"], dtype=float)
        lower = values - np.asarray(summary["minimum"], dtype=float)
        upper = np.asarray(summary["maximum"], dtype=float) - values
        ax.errorbar(
            HORIZON_LABELS,
            values,
            yerr=np.vstack([lower, upper]),
            marker=marker,
            ms=3.5,
            color=color,
            linestyle=linestyle,
            capsize=2,
            elinewidth=0.7,
            label=name,
        )
    ax.set_title(title, fontsize=7.5, pad=4)
    ax.set_ylabel("Hierarchical MAE (bpm)")
    ax.grid(axis="y", color="#E6E6E6", lw=0.5)
    panel_label(ax, label)


def extract_metric(frame: pd.DataFrame, **filters: object) -> list[float]:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    return selected.set_index("horizon_seconds").reindex(HORIZONS)["mae_bpm"].tolist()


def fixed_series(values: list[float]) -> dict[str, list[float]]:
    return {"median": values, "minimum": values, "maximum": values}


def q1_series(
    summary: pd.DataFrame,
    *,
    experiment: str,
    regime: str,
    mode: str,
) -> dict[str, list[float]]:
    selected = summary[
        (summary.experiment == experiment)
        & (summary.regime == regime)
        & (summary["mode"] == mode)
        & (summary.source_kind == "point")
        & (summary.metric == "mae_bpm")
    ].set_index("horizon_seconds").reindex(HORIZONS)
    if selected.value_median.isna().any():
        raise ValueError(f"Incomplete Q1 point series: {experiment}/{regime}/{mode}")
    return {
        "median": selected.value_median.tolist(),
        "minimum": selected.value_minimum.tolist(),
        "maximum": selected.value_maximum.tolist(),
    }


def independent_series(
    per_seed: pd.DataFrame,
    *,
    protocol: str,
    evaluation: str,
) -> dict[str, list[float]]:
    selected = per_seed[
        (per_seed.protocol == protocol) & (per_seed.evaluation == evaluation)
    ][["seed", "horizon_seconds", "independent_zero_mae_bpm"]].drop_duplicates()
    grouped = selected.groupby("horizon_seconds").independent_zero_mae_bpm
    summary = grouped.agg(["median", "min", "max"]).reindex(HORIZONS)
    if summary["median"].isna().any():
        raise ValueError(f"Incomplete independent-zero series: {protocol}/{evaluation}")
    return {
        "median": summary["median"].tolist(),
        "minimum": summary["min"].tolist(),
        "maximum": summary["max"].tolist(),
    }


def series_source_rows(regime: str, method: str, summary: dict[str, list[float]], seeds: str) -> list[dict]:
    return [
        {
            "regime": regime,
            "method": method,
            "horizon_seconds": horizon,
            "mae_median_bpm": summary["median"][index],
            "mae_minimum_bpm": summary["minimum"][index],
            "mae_maximum_bpm": summary["maximum"][index],
            "seed_scope": seeds,
        }
        for index, horizon in enumerate(HORIZONS)
    ]


def forest(ax: plt.Axes, frame: pd.DataFrame, labels: list[str], colors: list[str], panel: str,
           xlabel: str = "ΔMAE (bpm); negative favours first model") -> None:
    y = np.arange(len(frame))[::-1]
    for yi, (_, row), color in zip(y, frame.iterrows(), colors):
        ax.plot([row.ci_low_bpm, row.ci_high_bpm], [yi, yi], color=color, lw=1.4)
        ax.plot(row.delta_mae_bpm, yi, "o", color=color, ms=3.5)
    ax.axvline(0, color="#888888", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#EEEEEE", lw=0.5)
    panel_label(ax, panel)


def figure2() -> None:
    temporal = pd.read_csv(RESULTS / "temporal_uncertainty_point_v0_13_0.csv")
    temporal_base = pd.read_csv(RESULTS / "temporal_aligned_baselines_v0_13_0.csv")
    main = pd.read_csv(RESULTS / "uncertainty_point_metrics_v0_11_0.csv")
    naive = pd.read_csv(RESULTS / "naive_baseline_metrics_v0_5_0.csv")
    gru = pd.read_csv(RESULTS / "gru_user_generalization_metrics_v0_9_0.csv")
    paired = pd.read_csv(RESULTS / "paired_model_comparisons_v0_11_0.csv")
    temporal_paired = pd.read_csv(RESULTS / "temporal_paired_comparisons_v0_13_0.csv")

    temporal_data = {
        "History-informed": extract_metric(temporal, mode="history_informed"),
        "Zero-history": extract_metric(temporal, mode="zero_history"),
        "EWMA": extract_metric(temporal_base.rename(columns={"model": "method"}), method="ewma_alpha_0_1"),
        "Persistence": extract_metric(temporal_base.rename(columns={"model": "method"}), method="persistence"),
    }
    unseen_data = {
        "History-informed": extract_metric(main, regime="unseen_user_test", mode="history_informed"),
        "Zero-history": extract_metric(main, regime="unseen_user_test", mode="zero_history"),
        "EWMA": extract_metric(naive, regime="unseen_user_test", model="ewma"),
        "Persistence": extract_metric(naive, regime="unseen_user_test", model="persistence"),
    }
    external_data = {
        "Zero-history": extract_metric(main, regime="goldencheetah_frozen_external", mode="zero_history"),
        "GRU": extract_metric(gru, regime="goldencheetah_frozen_external", model="gru"),
        "EWMA": extract_metric(naive, regime="goldencheetah_frozen_external", model="ewma"),
        "Persistence": extract_metric(naive, regime="goldencheetah_frozen_external", model="persistence"),
    }
    source_rows = []
    for regime, payload in [("temporal", temporal_data), ("unseen_user", unseen_data), ("external", external_data)]:
        for method, values in payload.items():
            for horizon, value in zip(HORIZONS, values):
                source_rows.append({"regime": regime, "method": method, "horizon_seconds": horizon, "mae_bpm": value})
    SOURCE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(SOURCE / "Figure_2_point_forecast_source.csv", index=False)

    temp_effect = temporal_paired[temporal_paired.comparison_family == "temporal_history_vs_persistence"].copy()
    unseen_effect = paired[paired.comparison_family == "unseen_history_vs_zero"].copy()
    external_effect = paired[paired.comparison_family == "external_zero_vs_gru"].copy()
    effects = pd.concat([temp_effect, unseen_effect, external_effect], ignore_index=True)
    effects["group"] = np.repeat(["Temporal vs persistence", "Unseen-user history vs zero", "External zero vs GRU"], 3)
    effects.to_csv(SOURCE / "Figure_2_paired_effect_source.csv", index=False)
    labels = [f"{group} · {label}" for group, label in zip(effects.group, effects.horizon_seconds.map({60:"1 min",180:"3 min",300:"5 min"}))]
    effect_colors = [COLORS["temporal"]] * 3 + [COLORS["unseen"]] * 3 + [COLORS["external"]] * 3

    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    line_panel(axes[0], temporal_data, "Strict temporal test", "a")
    line_panel(axes[1], unseen_data, "Unseen-user test", "b")
    line_panel(axes[2], external_data, "Frozen external test", "c")
    for ax in axes:
        ax.set_ylim(5, 13)
    handles, names = axes[0].get_legend_handles_labels()
    h2, n2 = axes[2].get_legend_handles_labels()
    lookup = dict(zip(names + n2, handles + h2))
    fig.legend(lookup.values(), lookup.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=6.2)
    ax_forest = fig.add_subplot(gs[1, :])
    forest(ax_forest, effects, labels, effect_colors, "d")
    ax_forest.set_xlim(-1.55, 0.12)
    save(fig, "Figure_2_primary_performance")


def figure3() -> None:
    temporal_interval = pd.read_csv(RESULTS / "temporal_uncertainty_interval_v0_13_0.csv")
    main_interval = pd.read_csv(RESULTS / "uncertainty_interval_metrics_v0_11_0.csv")
    temporal_prob = pd.read_csv(RESULTS / "temporal_probabilistic_metrics_v0_13_0.csv")
    main_prob = pd.read_csv(RESULTS / "probabilistic_metrics_v0_11_0.csv")
    regimes = {
        "Temporal": (temporal_interval, "within_user_temporal_test", "history_informed", COLORS["temporal"]),
        "Unseen user": (main_interval, "unseen_user_test", "history_informed", COLORS["unseen"]),
        "External": (main_interval, "goldencheetah_frozen_external", "zero_history", COLORS["external"]),
    }
    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.95])
    coverage_sources = []
    for panel, (name, (frame, regime, mode, color)) in enumerate(regimes.items()):
        ax = fig.add_subplot(gs[0, panel])
        selected = frame[(frame.regime == regime) & (frame["mode"] == mode) & (frame.calibrated == True)]
        for horizon, marker in zip(HORIZONS, ["o", "s", "^"]):
            row = selected[selected.horizon_seconds == horizon].sort_values("nominal_coverage")
            ax.plot(row.nominal_coverage, row.picp, marker=marker, ms=3.5, label={60:"1 min",180:"3 min",300:"5 min"}[horizon], color=color, alpha={60:1.0,180:0.75,300:0.5}[horizon])
            tmp = row.copy(); tmp["display_regime"] = name; coverage_sources.append(tmp)
        ax.plot([0.45, 0.93], [0.45, 0.93], color="#888888", ls="--", lw=0.8)
        ax.set_xlim(0.46, 0.92); ax.set_ylim(0.43, 0.95)
        ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage")
        ax.set_title(name, fontsize=7.5)
        ax.grid(color="#EEEEEE", lw=0.5)
        panel_label(ax, chr(ord("a") + panel))
        if panel == 0: ax.legend(fontsize=6, loc="lower right")
    pd.concat(coverage_sources).to_csv(SOURCE / "Figure_4_coverage_source.csv", index=False)

    ax_width = fig.add_subplot(gs[1, 0])
    ax_wis = fig.add_subplot(gs[1, 1])
    ax_rho = fig.add_subplot(gs[1, 2])
    diagnostic_rows = []
    for name, (frame, regime, mode, color) in regimes.items():
        chosen = frame[(frame.regime == regime) & (frame["mode"] == mode) & (frame.calibrated == True) & np.isclose(frame.nominal_coverage, 0.9)].sort_values("horizon_seconds")
        ax_width.plot(HORIZON_LABELS, chosen.mean_interval_width_bpm, "o-", color=color, ms=3.5, label=name)
        prob_frame = temporal_prob if name == "Temporal" else main_prob
        prob_regime = "within_user_temporal_test" if name == "Temporal" else regime
        p = prob_frame[(prob_frame.regime == prob_regime) & (prob_frame["mode"] == mode) & (prob_frame.calibrated == True)].sort_values("horizon_seconds")
        ax_wis.plot(HORIZON_LABELS, p.weighted_interval_score, "o-", color=color, ms=3.5)
        ax_rho.plot(HORIZON_LABELS, p.mean_user_spearman_width_absolute_error, "o-", color=color, ms=3.5)
        p2 = p.copy(); p2["display_regime"] = name; diagnostic_rows.append(p2)
    pd.concat(diagnostic_rows).to_csv(SOURCE / "Figure_4_probabilistic_source.csv", index=False)
    ax_width.set_ylabel("90% interval width (bpm)")
    ax_wis.set_ylabel("Weighted interval score")
    ax_rho.set_ylabel("Width–error Spearman ρ")
    for ax, label in zip([ax_width, ax_wis, ax_rho], ["d", "e", "f"]):
        ax.grid(axis="y", color="#EEEEEE", lw=0.5); panel_label(ax, label)
    ax_width.legend(fontsize=6)
    save(fig, "Figure_4_uncertainty_calibration")


def annotated_heatmap(ax: plt.Axes, matrix: np.ndarray, row_labels: list[str], title: str,
                      vmin: float, vmax: float, panel: str, cmap: str = "Blues") -> mpl.image.AxesImage:
    color_map = mpl.colormaps[cmap].copy()
    color_map.set_bad("#F0F0F0")
    image = ax.imshow(matrix, aspect="auto", cmap=color_map, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(3), HORIZON_LABELS)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, fontsize=7.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "NR", ha="center", va="center", fontsize=6, color="#555555")
                continue
            norm = (value - vmin) / max(vmax - vmin, 1e-9)
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6, color="white" if norm > 0.58 else "black")
    for spine in ax.spines.values(): spine.set_visible(False)
    panel_label(ax, panel)
    return image


def figure4() -> None:
    sport = pd.read_csv(RESULTS / "sport_shift_point_v0_12_0.csv")
    baseline = pd.read_csv(RESULTS / "sport_shift_aligned_baselines_v0_12_0.csv")
    families = ["outdoor_cycling", "indoor_virtual_cycling", "running", "walking_hiking", "strength_cross_training"]
    labels = ["Outdoor cycling", "Indoor/virtual", "Running", "Walking/hiking", "Strength/cross"]
    matrices = {}
    for kind in ["unseen_sport", "joint_user_sport"]:
        matrix = np.empty((len(families), 3))
        for i, family in enumerate(families):
            regime = f"{kind}__{family}"
            row = sport[(sport.held_sport_family == family) & (sport.regime == regime) & (sport["mode"] == "history_informed")].set_index("horizon_seconds")
            matrix[i] = row.reindex(HORIZONS).mae_bpm
        matrices[kind] = matrix
    delta = np.empty((len(families), 3))
    for i, family in enumerate(families):
        regime = f"unseen_sport__{family}"
        model = sport[(sport.held_sport_family == family) & (sport.regime == regime) & (sport["mode"] == "history_informed")].set_index("horizon_seconds")
        base = baseline[(baseline.held_sport_family == family) & (baseline.regime == regime) & (baseline.model == "ewma_alpha_0_1")].set_index("horizon_seconds")
        delta[i] = model.reindex(HORIZONS).mae_bpm - base.reindex(HORIZONS).mae_bpm
    source = []
    for i, family in enumerate(families):
        for j, horizon in enumerate(HORIZONS):
            source.append({"family": family, "horizon_seconds": horizon, "same_user_mae": matrices['unseen_sport'][i,j], "joint_shift_mae": matrices['joint_user_sport'][i,j], "delta_vs_ewma": delta[i,j]})
    pd.DataFrame(source).to_csv(SOURCE / "Figure_3_sport_shift_source.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    image = annotated_heatmap(ax_a, matrices["unseen_sport"], labels, "Seen users, unseen sport", 5, 13, "a")
    annotated_heatmap(ax_b, matrices["joint_user_sport"], labels, "Unseen users + unseen sport", 5, 13, "b")
    cbar = fig.colorbar(image, ax=[ax_a, ax_b], shrink=0.65, pad=0.02)
    cbar.set_label("Hierarchical MAE (bpm)")
    max_abs = max(abs(delta.min()), abs(delta.max()))
    im_delta = ax_c.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    ax_c.set_xticks(range(3), HORIZON_LABELS); ax_c.set_yticks(range(len(labels)), labels)
    ax_c.set_title("Model − EWMA MAE (seen users)", fontsize=7.5)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax_c.text(j, i, f"{delta[i,j]:+.1f}", ha="center", va="center", fontsize=6)
    for spine in ax_c.spines.values(): spine.set_visible(False)
    panel_label(ax_c, "c")
    cb2 = fig.colorbar(im_delta, ax=ax_c, shrink=0.72, pad=0.02); cb2.set_label("ΔMAE (bpm)")
    support = sport.groupby(["held_sport_family", "regime"])[["users", "sessions"]].first().reset_index()
    same_users=[]; joint_users=[]
    for family in families:
        same_users.append(int(support[(support.held_sport_family==family)&(support.regime==f'unseen_sport__{family}')].users.iloc[0]))
        joint_users.append(int(support[(support.held_sport_family==family)&(support.regime==f'joint_user_sport__{family}')].users.iloc[0]))
    x=np.arange(len(labels)); w=.36
    ax_d.bar(x-w/2,same_users,w,color=COLORS['history'],label='Seen users')
    ax_d.bar(x+w/2,joint_users,w,color=COLORS['joint'],label='Joint shift')
    ax_d.axhline(25,color=COLORS['external'],ls='--',lw=.8,label='25-user caution')
    ax_d.set_yscale('log'); ax_d.set_ylabel('Users (log scale)'); ax_d.set_xticks(x, ['Outdoor','Indoor','Run','Walk','Strength'],rotation=25,ha='right')
    ax_d.legend(fontsize=6,ncol=2); ax_d.grid(axis='y',color='#EEEEEE',lw=.5); panel_label(ax_d,'d')
    pd.DataFrame({'family':families,'seen_user_sport_shift_users':same_users,'joint_shift_users':joint_users}).to_csv(SOURCE/'Figure_3_support_source.csv',index=False)
    save(fig, "Figure_3_sport_shift")


def supplementary_figure1() -> None:
    signal = pd.read_csv(RESULTS / "signal_ablation_paired_v0_14_0.csv")
    dense = pd.read_csv(RESULTS / "dense_origin_point_v0_15_0.csv")
    standard = pd.read_csv(RESULTS / "uncertainty_point_metrics_v0_11_0.csv")
    temporal_effect = pd.read_csv(RESULTS / "temporal_paired_comparisons_v0_13_0.csv")
    unseen_effect = pd.read_csv(RESULTS / "paired_model_comparisons_v0_11_0.csv")
    gender = pd.read_csv(RESULTS / "recorded_gender_differences_v0_16_0.csv")
    fig = plt.figure(figsize=(7.2, 5.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]

    def signal_label(row: pd.Series) -> str:
        family = str(row.comparison_family)
        if family.startswith('unseen_history'):
            regime = 'Unseen/history'
        elif family.startswith('unseen_zero'):
            regime = 'Unseen/zero'
        else:
            regime = 'External/zero'
        return f"{regime} · {int(row.horizon_seconds/60)} min"

    sig_labels=[signal_label(r) for _,r in signal.iterrows()]
    sig_colors=[COLORS['unseen']]*6+[COLORS['external']]*3
    forest(axes[0],signal,sig_labels,sig_colors,'a','Multimodal − HR-only ΔMAE (bpm)')
    axes[0].set_title('Signal ablation',fontsize=7.5)

    std=standard[standard.regime=='unseen_user_test'][['mode','horizon_seconds','mae_bpm']].rename(columns={'mae_bpm':'standard'})
    den=dense[['mode','horizon_seconds','mae_bpm']].rename(columns={'mae_bpm':'dense'})
    stride=std.merge(den);stride['delta']=stride.dense-stride.standard
    for mode,color,marker in [('history_informed',COLORS['history'],'o'),('zero_history',COLORS['zero'],'s')]:
        r=stride[stride['mode']==mode].sort_values('horizon_seconds');axes[1].plot(HORIZON_LABELS,r.delta,marker=marker,color=color,label=mode.replace('_',' '))
    axes[1].axhline(0,color='#888888',ls='--',lw=.8);axes[1].set_ylabel('Dense 60-s − standard 300-s MAE');axes[1].set_title('Evaluation-origin sensitivity',fontsize=7.5);axes[1].legend(fontsize=6);axes[1].grid(axis='y',color='#EEEEEE',lw=.5);panel_label(axes[1],'b')

    history_effect=pd.concat([temporal_effect[temporal_effect.comparison_family=='temporal_history_vs_zero'],unseen_effect[unseen_effect.comparison_family=='unseen_history_vs_zero']],ignore_index=True)
    hist_labels=[f"{('Temporal' if 'temporal' in r.comparison_family else 'Unseen user')} · {int(r.horizon_seconds/60)} min" for _,r in history_effect.iterrows()]
    forest(axes[2],history_effect,hist_labels,[COLORS['temporal']]*3+[COLORS['unseen']]*3,'c','History − zero-history ΔMAE (bpm)')
    axes[2].set_title('Personalization ablation',fontsize=7.5)

    gen=gender[gender['mode']=='history_informed'].copy()
    gen_labels=[f"{('Temporal' if r.regime.startswith('within') else 'Unseen user')} · {int(r.horizon_seconds/60)} min" for _,r in gen.iterrows()]
    gen_colors=[COLORS['unseen'] if r.support_status=='supported_descriptive' else COLORS['gold'] for _,r in gen.iterrows()]
    forest(axes[3],gen.rename(columns={'delta_mae_bpm':'delta_mae_bpm'}),gen_labels,gen_colors,'d','Female − male descriptive ΔMAE (bpm)')
    axes[3].set_title('Recorded-gender subgroup',fontsize=7.5)
    signal.to_csv(SOURCE/'Supplementary_Figure_1_signal_source.csv',index=False);stride.to_csv(SOURCE/'Supplementary_Figure_1_stride_source.csv',index=False);history_effect.to_csv(SOURCE/'Supplementary_Figure_1_history_source.csv',index=False);gen.to_csv(SOURCE/'Supplementary_Figure_1_gender_source.csv',index=False)
    save(fig, "Supplementary_Figure_1_ablation_sensitivity")


def graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(13.28, 5.31))
    ax.set_xlim(0, 13.28)
    ax.set_ylim(0, 5.31)
    ax.set_axis_off()

    ax.text(
        6.64,
        5.02,
        "Leakage-controlled exercise heart-rate forecasting",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["black"],
    )

    # Causal inputs
    box(ax, (0.25, 2.15), 2.55, 2.35, "", "#EAF1F8", COLORS["history"])
    ax.text(1.525, 4.20, "CAUSAL INPUT", ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["history"])
    x = np.linspace(0.55, 2.48, 80)
    ax.plot(x, 3.55 + 0.20 * np.sin(np.linspace(0, 3.6 * np.pi, 80)) + np.linspace(0, 0.25, 80), color=COLORS["external"], lw=2)
    ax.text(0.47, 3.52, "HR", ha="left", va="center", fontsize=8, color=COLORS["external"], fontweight="bold")
    ax.plot(x, 3.12 + 0.10 * np.sin(np.linspace(0, 5 * np.pi, 80)), color=COLORS["temporal"], lw=1.7)
    ax.text(0.47, 3.10, "Speed", ha="left", va="center", fontsize=7.5, color=COLORS["temporal"], fontweight="bold")
    ax.plot(x, 2.70 + np.linspace(-0.10, 0.14, 80) + 0.06 * np.sin(np.linspace(0, 2 * np.pi, 80)), color=COLORS["gold"], lw=1.7)
    ax.text(0.47, 2.67, "Altitude", ha="left", va="center", fontsize=7.5, color=COLORS["gold"], fontweight="bold")
    ax.text(1.525, 2.38, "Previous 5 min only", ha="center", va="center", fontsize=8.5)
    box(ax, (0.25, 1.28), 2.55, 0.58, "Completed prior workouts only", "#E3F0ED", COLORS["temporal"], fontsize=8)

    # Model
    arrow(ax, (2.92, 3.18), (3.55, 3.18), COLORS["black"])
    arrow(ax, (2.92, 1.57), (3.55, 2.35), COLORS["black"])
    box(ax, (3.62, 1.85), 2.45, 2.65, "", "#EEE8F2", COLORS["joint"])
    ax.text(4.845, 4.18, "HISTORY-QUANTILE TCN", ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["joint"])
    box(ax, (3.95, 3.28), 1.79, 0.55, "Causal TCN", "#DDE8F3", COLORS["history"], fontsize=8)
    box(ax, (3.95, 2.55), 1.79, 0.55, "History encoder", "#E3F0ED", COLORS["temporal"], fontsize=8)
    box(ax, (3.95, 1.98), 1.79, 0.42, "7 ordered quantiles", "#F5DEDD", COLORS["external"], fontsize=7.5)
    ax.text(4.845, 1.54, "1 / 3 / 5-min HR forecasts", ha="center", va="center", fontsize=8.5, color=COLORS["black"], fontweight="bold")

    # Shift protocols
    arrow(ax, (6.20, 3.18), (6.78, 3.18), COLORS["black"])
    box(ax, (6.85, 1.85), 2.58, 2.65, "", "#F5F5F5", "#777777")
    ax.text(8.14, 4.18, "SEPARATE SHIFT TESTS", ha="center", va="center", fontsize=10, fontweight="bold")
    rows = [
        ("Later sessions", COLORS["temporal"]),
        ("New users", COLORS["unseen"]),
        ("Held-out sport", COLORS["gold"]),
        ("New user + sport", COLORS["joint"]),
        ("GoldenCheetah external", COLORS["external"]),
    ]
    for idx, (label, color) in enumerate(rows):
        y = 3.67 - idx * 0.43
        ax.add_patch(patches.Circle((7.18, y), 0.08, facecolor=color, edgecolor="none"))
        ax.text(7.38, y, label, ha="left", va="center", fontsize=8)

    # Results
    arrow(ax, (9.55, 3.18), (10.05, 3.18), COLORS["black"])
    box(ax, (10.12, 1.85), 2.91, 2.65, "", "#FFF8F1", COLORS["external"])
    ax.text(11.575, 4.18, "WHAT TRANSFERS?", ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["external"])
    ax.text(10.38, 3.72, "History gain vs independent zero", ha="left", va="center", fontsize=7.3, fontweight="bold")
    ax.text(12.77, 3.43, "0.03–0.11 bpm", ha="right", va="center", fontsize=9.5, fontweight="bold", color=COLORS["history"])
    ax.text(10.38, 3.08, "External MAE (1 / 3 / 5 min)", ha="left", va="center", fontsize=7.6, fontweight="bold")
    ax.text(12.77, 2.82, "7.47 / 10.21 / 11.21", ha="right", va="center", fontsize=9.5, fontweight="bold", color=COLORS["external"])
    ax.text(10.38, 2.46, "External 90% PICP (1 / 3 / 5 min)", ha="left", va="center", fontsize=7.6, fontweight="bold")
    ax.text(12.77, 2.20, "0.880 / 0.859 / 0.850", ha="right", va="center", fontsize=9.5, fontweight="bold", color=COLORS["external"])
    ax.text(11.575, 1.96, "Coverage degrades despite wider intervals", ha="center", va="center", fontsize=7.5, color=COLORS["external"])

    ax.add_patch(patches.FancyBboxPatch((0.25, 0.30), 12.78, 0.62, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=COLORS["black"], edgecolor="none"))
    ax.text(
        6.64,
        0.61,
        "Personalization is incremental; sport and dataset shifts remain the dominant reliability constraints.",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    save(fig, "Graphical_Abstract")


def figure4_multiseed(
    *, reporting_threshold: int | None = None, output_stem: str = "Figure_3_sport_shift"
) -> None:
    q1 = pd.read_csv(Q1_AGGREGATION / "seed_variability_summary_v0_22_0.csv")
    baseline = pd.read_csv(RESULTS / "sport_shift_aligned_baselines_v0_12_0.csv")
    sport = q1[
        (q1.experiment == "held_sport")
        & (q1.source_kind == "point")
        & (q1.metric == "mae_bpm")
        & (q1["mode"] == "history_informed")
    ].copy()
    families = [
        "outdoor_cycling",
        "indoor_virtual_cycling",
        "running",
        "walking_hiking",
        "strength_cross_training",
    ]
    labels = [
        "Outdoor cycling",
        "Indoor/virtual",
        "Running",
        "Walking/hiking",
        "Strength/cross",
    ]
    matrices: dict[str, np.ndarray] = {}
    minima: dict[str, np.ndarray] = {}
    maxima: dict[str, np.ndarray] = {}
    for kind in ["unseen_sport", "joint_user_sport"]:
        matrix = np.empty((len(families), 3))
        min_matrix = np.empty_like(matrix)
        max_matrix = np.empty_like(matrix)
        for i, family in enumerate(families):
            regime = f"{kind}__{family}"
            row = sport[
                (sport.family == family) & (sport.regime == regime)
            ].set_index("horizon_seconds").reindex(HORIZONS)
            matrix[i] = row.value_median
            min_matrix[i] = row.value_minimum
            max_matrix[i] = row.value_maximum
        matrices[kind] = matrix
        minima[kind] = min_matrix
        maxima[kind] = max_matrix

    delta = np.empty((len(families), 3))
    for i, family in enumerate(families):
        regime = f"unseen_sport__{family}"
        base = baseline[
            (baseline.held_sport_family == family)
            & (baseline.regime == regime)
            & (baseline.model == "ewma_alpha_0_1")
        ].set_index("horizon_seconds").reindex(HORIZONS)
        delta[i] = matrices["unseen_sport"][i] - base.mae_bpm.to_numpy()

    support = (
        sport.groupby(["family", "regime"])[["users", "sessions"]]
        .first()
        .reset_index()
    )
    same_users: list[int] = []
    joint_users: list[int] = []
    for family in families:
        same_users.append(
            int(
                support[
                    (support.family == family)
                    & (support.regime == f"unseen_sport__{family}")
                ].users.iloc[0]
            )
        )
        joint_users.append(
            int(
                support[
                    (support.family == family)
                    & (support.regime == f"joint_user_sport__{family}")
                ].users.iloc[0]
            )
        )

    source: list[dict] = []
    for i, family in enumerate(families):
        for j, horizon in enumerate(HORIZONS):
            report_joint = reporting_threshold is None or joint_users[i] >= reporting_threshold
            source.append(
                {
                    "family": family,
                    "horizon_seconds": horizon,
                    "same_user_mae_median_bpm": matrices["unseen_sport"][i, j],
                    "same_user_mae_minimum_bpm": minima["unseen_sport"][i, j],
                    "same_user_mae_maximum_bpm": maxima["unseen_sport"][i, j],
                    "joint_shift_mae_median_bpm": matrices["joint_user_sport"][i, j] if report_joint else np.nan,
                    "joint_shift_mae_minimum_bpm": minima["joint_user_sport"][i, j] if report_joint else np.nan,
                    "joint_shift_mae_maximum_bpm": maxima["joint_user_sport"][i, j] if report_joint else np.nan,
                    "joint_shift_users": joint_users[i],
                    "journal_reporting_threshold_users": reporting_threshold,
                    "joint_shift_outcome_reported": report_joint,
                    "median_model_minus_deterministic_ewma_bpm": delta[i, j],
                    "seed_scope": "3-seed median and range",
                }
            )
    pd.DataFrame(source).to_csv(
        SOURCE / f"{output_stem}_source.csv", index=False
    )

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    ax_a, ax_b, ax_c, ax_d = [
        fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)
    ]
    image = annotated_heatmap(
        ax_a,
        matrices["unseen_sport"],
        labels,
        "Seen users, unseen sport (3-seed median)",
        5,
        13,
        "a",
    )
    joint_display = matrices["joint_user_sport"].copy()
    if reporting_threshold is not None:
        for i, users in enumerate(joint_users):
            if users < reporting_threshold:
                joint_display[i, :] = np.nan
    joint_title = "Unseen users + unseen sport (3-seed median)"
    if reporting_threshold is not None:
        joint_title += f"; n >= {reporting_threshold}"
    annotated_heatmap(
        ax_b,
        joint_display,
        labels,
        joint_title,
        5,
        13,
        "b",
    )
    cbar = fig.colorbar(image, ax=[ax_a, ax_b], shrink=0.65, pad=0.02)
    cbar.set_label("Hierarchical MAE (bpm)")
    max_abs = max(abs(delta.min()), abs(delta.max()))
    im_delta = ax_c.imshow(
        delta, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs
    )
    ax_c.set_xticks(range(3), HORIZON_LABELS)
    ax_c.set_yticks(range(len(labels)), labels)
    ax_c.set_title("Median model − deterministic EWMA", fontsize=7.5)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax_c.text(
                j,
                i,
                f"{delta[i, j]:+.1f}",
                ha="center",
                va="center",
                fontsize=6,
            )
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    panel_label(ax_c, "c")
    cb2 = fig.colorbar(im_delta, ax=ax_c, shrink=0.72, pad=0.02)
    cb2.set_label("ΔMAE (bpm)")

    x = np.arange(len(labels))
    width = 0.36
    ax_d.bar(
        x - width / 2,
        same_users,
        width,
        color=COLORS["history"],
        label="Seen users",
    )
    ax_d.bar(
        x + width / 2,
        joint_users,
        width,
        color=COLORS["joint"],
        label="Joint shift",
    )
    threshold = reporting_threshold if reporting_threshold is not None else 25
    threshold_label = (
        f"{threshold}-user reporting threshold"
        if reporting_threshold is not None
        else "25-user caution"
    )
    ax_d.axhline(
        threshold, color=COLORS["external"], ls="--", lw=0.8, label=threshold_label
    )
    ax_d.set_yscale("log")
    ax_d.set_ylabel("Users (log scale)")
    ax_d.set_xticks(
        x,
        ["Outdoor", "Indoor", "Run", "Walk", "Strength"],
        rotation=25,
        ha="right",
    )
    ax_d.legend(fontsize=6, ncol=2)
    ax_d.grid(axis="y", color="#EEEEEE", lw=0.5)
    panel_label(ax_d, "d")
    support_output_stem = (
        "Figure_3_support"
        if output_stem == "Figure_3_sport_shift"
        else f"{output_stem}_support"
    )
    pd.DataFrame(
        {
            "family": families,
            "seen_user_sport_shift_users": same_users,
            "joint_shift_users": joint_users,
        }
    ).to_csv(SOURCE / f"{support_output_stem}_source.csv", index=False)
    save(fig, output_stem)


def supplementary_figure1_multiseed(
    *, include_gender: bool = True,
    output_stem: str = "Supplementary_Figure_1_ablation_sensitivity",
) -> None:
    signal = pd.read_csv(RESULTS / "signal_ablation_paired_v0_14_0.csv")
    dense = pd.read_csv(RESULTS / "dense_origin_point_v0_15_0.csv")
    standard = pd.read_csv(RESULTS / "uncertainty_point_metrics_v0_11_0.csv")
    gender = (
        pd.read_csv(RESULTS / "recorded_gender_differences_v0_16_0.csv")
        if include_gender
        else None
    )
    history_effect = pd.read_csv(
        ZERO_AGGREGATION / "strategy_contrast_user_bootstrap_v0_23_0.csv"
    )
    history_effect = history_effect[
        (history_effect.evaluation == "internal_test")
        & (history_effect.contrast == "mixed_history_minus_independent_zero")
    ].copy()
    history_effect = history_effect.rename(
        columns={
            "estimate_bpm": "delta_mae_bpm",
            "percentile_95_ci_low_bpm": "ci_low_bpm",
            "percentile_95_ci_high_bpm": "ci_high_bpm",
        }
    )

    if include_gender:
        fig = plt.figure(figsize=(7.2, 5.7), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    else:
        fig = plt.figure(figsize=(7.2, 7.2), constrained_layout=True)
        gs = fig.add_gridspec(3, 1)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    def signal_label(row: pd.Series) -> str:
        family = str(row.comparison_family)
        if family.startswith("unseen_history"):
            regime = "Unseen/history-informed"
        elif family.startswith("unseen_zero"):
            regime = "Unseen/history-masked"
        else:
            regime = "Cross-source/history-masked"
        return f"{regime} · {int(row.horizon_seconds / 60)} min"

    sig_labels = [signal_label(row) for _, row in signal.iterrows()]
    sig_colors = [COLORS["unseen"]] * 6 + [COLORS["external"]] * 3
    forest(
        axes[0],
        signal,
        sig_labels,
        sig_colors,
        "a",
        "Multimodal − HR-only ΔMAE (bpm)",
    )
    axes[0].set_title("Signal ablation (reference seed)", fontsize=7.5)

    standard_rows = standard[standard.regime == "unseen_user_test"][
        ["mode", "horizon_seconds", "mae_bpm"]
    ].rename(columns={"mae_bpm": "standard"})
    dense_rows = dense[["mode", "horizon_seconds", "mae_bpm"]].rename(
        columns={"mae_bpm": "dense"}
    )
    stride = standard_rows.merge(dense_rows)
    stride["delta"] = stride.dense - stride.standard
    for mode, color, marker in [
        ("history_informed", COLORS["history"], "o"),
        ("zero_history", COLORS["zero"], "s"),
    ]:
        row = stride[stride["mode"] == mode].sort_values("horizon_seconds")
        axes[1].plot(
            HORIZON_LABELS,
            row.delta,
            marker=marker,
            color=color,
            label={
                "history_informed": "history-informed",
                "zero_history": "history-masked",
            }[mode],
        )
    axes[1].axhline(0, color="#888888", ls="--", lw=0.8)
    axes[1].set_ylabel("Dense 60-s − standard 300-s MAE")
    axes[1].set_title("Origin sensitivity (reference seed)", fontsize=7.5)
    axes[1].legend(fontsize=6)
    axes[1].grid(axis="y", color="#EEEEEE", lw=0.5)
    panel_label(axes[1], "b")

    history_effect["display_regime"] = history_effect.protocol.map(
        {"strict_temporal": "Temporal", "unseen_user": "Unseen user"}
    )
    hist_labels = [
        f"{row.display_regime} · {int(row.horizon_seconds / 60)} min"
        for _, row in history_effect.iterrows()
    ]
    forest(
        axes[2],
        history_effect,
        hist_labels,
        [COLORS["temporal"]] * 3 + [COLORS["unseen"]] * 3,
        "c",
        "History-informed − zero-history-trained ΔMAE (bpm)",
    )
    axes[2].set_title("Zero-history-trained strategy (5 seeds)", fontsize=7.5)

    if not include_gender:
        signal.assign(seed_scope="reference seed 20260722").to_csv(
            SOURCE / "Supplementary_Figure_1_signal_source.csv", index=False
        )
        stride.assign(seed_scope="reference seed 20260722").to_csv(
            SOURCE / "Supplementary_Figure_1_stride_source.csv", index=False
        )
        history_effect.assign(
            seed_scope="paired per-user difference averaged over 5 seeds"
        ).to_csv(SOURCE / "Supplementary_Figure_1_history_source.csv", index=False)
        save(fig, output_stem)
        return

    assert gender is not None
    gen = gender[gender["mode"] == "history_informed"].copy()
    gen_labels = [
        f"{('Temporal' if row.regime.startswith('within') else 'Unseen user')} · "
        f"{int(row.horizon_seconds / 60)} min"
        for _, row in gen.iterrows()
    ]
    gen_colors = [
        COLORS["unseen"]
        if row.support_status == "supported_descriptive"
        else COLORS["gold"]
        for _, row in gen.iterrows()
    ]
    forest(
        axes[3],
        gen,
        gen_labels,
        gen_colors,
        "d",
        "Female − male descriptive ΔMAE (bpm)",
    )
    axes[3].set_title("Recorded gender (reference seed)", fontsize=7.5)

    signal.assign(seed_scope="reference seed 20260722").to_csv(
        SOURCE / "Supplementary_Figure_1_signal_source.csv", index=False
    )
    stride.assign(seed_scope="reference seed 20260722").to_csv(
        SOURCE / "Supplementary_Figure_1_stride_source.csv", index=False
    )
    history_effect.assign(
        seed_scope="paired per-user difference averaged over 5 seeds"
    ).to_csv(SOURCE / "Supplementary_Figure_1_history_source.csv", index=False)
    gen.assign(seed_scope="reference seed 20260722").to_csv(
        SOURCE / "Supplementary_Figure_1_gender_source.csv", index=False
    )
    save(fig, output_stem)


def figure3_multiseed() -> None:
    q1 = pd.read_csv(Q1_AGGREGATION / "seed_variability_summary_v0_22_0.csv")
    temporal_prob = pd.read_csv(RESULTS / "temporal_probabilistic_metrics_v0_13_0.csv")
    main_prob = pd.read_csv(RESULTS / "probabilistic_metrics_v0_11_0.csv")
    regimes = {
        "Temporal · history-informed": (
            "temporal_main",
            "within_user_temporal_test",
            "history_informed",
            COLORS["temporal"],
        ),
        "Unseen user · history-masked": (
            "unseen_main",
            "unseen_user_test",
            "zero_history",
            COLORS["unseen"],
        ),
        "Cross-source · history-masked": (
            "unseen_main",
            "goldencheetah_frozen_external",
            "zero_history",
            COLORS["external"],
        ),
    }

    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.95])
    coverage_sources: list[pd.DataFrame] = []
    for panel, (name, (experiment, regime, mode, color)) in enumerate(regimes.items()):
        ax = fig.add_subplot(gs[0, panel])
        selected = q1[
            (q1.experiment == experiment)
            & (q1.regime == regime)
            & (q1["mode"] == mode)
            & (q1.source_kind == "interval")
            & (q1.metric == "picp")
            & (q1.calibrated == True)
        ].copy()
        for horizon, marker, alpha in zip(HORIZONS, ["o", "s", "^"], [1.0, 0.75, 0.5]):
            row = selected[selected.horizon_seconds == horizon].sort_values(
                "nominal_coverage"
            )
            y = row.value_median.to_numpy()
            yerr = np.vstack(
                [
                    y - row.value_minimum.to_numpy(),
                    row.value_maximum.to_numpy() - y,
                ]
            )
            ax.errorbar(
                row.nominal_coverage,
                y,
                yerr=yerr,
                marker=marker,
                ms=3.5,
                capsize=2,
                elinewidth=0.7,
                label={60: "1 min", 180: "3 min", 300: "5 min"}[horizon],
                color=color,
                alpha=alpha,
            )
            tmp = row.copy()
            tmp["display_regime"] = name
            tmp["seed_scope"] = "5-seed median and range"
            coverage_sources.append(tmp)
        ax.plot([0.45, 0.93], [0.45, 0.93], color="#888888", ls="--", lw=0.8)
        ax.set_xlim(0.46, 0.92)
        ax.set_ylim(0.43, 0.95)
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Observed coverage")
        ax.set_title(name, fontsize=7.5)
        ax.grid(color="#EEEEEE", lw=0.5)
        panel_label(ax, chr(ord("a") + panel))
        if panel == 0:
            ax.legend(fontsize=6, loc="lower right")
    pd.concat(coverage_sources).to_csv(
        SOURCE / "Figure_4_coverage_source.csv", index=False
    )

    ax_width = fig.add_subplot(gs[1, 0])
    ax_wis = fig.add_subplot(gs[1, 1])
    ax_rho = fig.add_subplot(gs[1, 2])
    width_sources: list[pd.DataFrame] = []
    diagnostic_rows: list[pd.DataFrame] = []
    for name, (experiment, regime, mode, color) in regimes.items():
        chosen = q1[
            (q1.experiment == experiment)
            & (q1.regime == regime)
            & (q1["mode"] == mode)
            & (q1.source_kind == "interval")
            & (q1.metric == "mean_interval_width_bpm")
            & (q1.calibrated == True)
            & np.isclose(q1.nominal_coverage, 0.9)
        ].sort_values("horizon_seconds")
        y = chosen.value_median.to_numpy()
        ax_width.errorbar(
            HORIZON_LABELS,
            y,
            yerr=np.vstack(
                [
                    y - chosen.value_minimum.to_numpy(),
                    chosen.value_maximum.to_numpy() - y,
                ]
            ),
            marker="o",
            color=color,
            ms=3.5,
            capsize=2,
            elinewidth=0.7,
            label=name,
        )
        width_source = chosen.copy()
        width_source["display_regime"] = name
        width_source["seed_scope"] = "5-seed median and range"
        width_sources.append(width_source)

        prob_frame = temporal_prob if name.startswith("Temporal") else main_prob
        prob_regime = "within_user_temporal_test" if name.startswith("Temporal") else regime
        p = prob_frame[
            (prob_frame.regime == prob_regime)
            & (prob_frame["mode"] == mode)
            & (prob_frame.calibrated == True)
        ].sort_values("horizon_seconds")
        ax_wis.plot(
            HORIZON_LABELS,
            p.weighted_interval_score,
            "o-",
            color=color,
            ms=3.5,
        )
        ax_rho.plot(
            HORIZON_LABELS,
            p.mean_user_spearman_width_absolute_error,
            "o-",
            color=color,
            ms=3.5,
        )
        p2 = p.copy()
        p2["display_regime"] = name
        p2["seed_scope"] = "reference seed 20260722"
        diagnostic_rows.append(p2)

    pd.concat(width_sources).to_csv(
        SOURCE / "Figure_4_width_source.csv", index=False
    )
    pd.concat(diagnostic_rows).to_csv(
        SOURCE / "Figure_4_probabilistic_source.csv", index=False
    )
    ax_width.set_ylabel("90% interval width (bpm)")
    ax_wis.set_ylabel("WIS (lower is better)")
    ax_rho.set_ylabel("Width–error Spearman rho")
    ax_wis.set_title("Reference seed", fontsize=6.5)
    ax_rho.set_title("Reference seed", fontsize=6.5)
    for ax, label in zip([ax_width, ax_wis, ax_rho], ["d", "e", "f"]):
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
        panel_label(ax, label)
    ax_width.legend(fontsize=6)
    save(fig, "Figure_4_uncertainty_calibration")


def figure2_multiseed() -> None:
    q1 = pd.read_csv(Q1_AGGREGATION / "seed_variability_summary_v0_22_0.csv")
    independent = pd.read_csv(
        ZERO_AGGREGATION / "strategy_contrasts_per_seed_v0_23_0.csv"
    )
    effects = pd.read_csv(
        ZERO_AGGREGATION / "strategy_contrast_user_bootstrap_v0_23_0.csv"
    )
    temporal_base = pd.read_csv(RESULTS / "temporal_aligned_baselines_v0_13_0.csv")
    naive = pd.read_csv(RESULTS / "naive_baseline_metrics_v0_5_0.csv")

    temporal_data = {
        "History-informed": q1_series(
            q1,
            experiment="temporal_main",
            regime="within_user_temporal_test",
            mode="history_informed",
        ),
        "Zero-history-trained": independent_series(
            independent, protocol="strict_temporal", evaluation="internal_test"
        ),
        "GRU": q1_series(
            q1,
            experiment="temporal_gru",
            regime="within_user_temporal_test",
            mode="not_applicable",
        ),
        "EWMA": fixed_series(
            extract_metric(
                temporal_base.rename(columns={"model": "method"}),
                method="ewma_alpha_0_1",
            )
        ),
        "Persistence": fixed_series(
            extract_metric(
                temporal_base.rename(columns={"model": "method"}),
                method="persistence",
            )
        ),
    }
    unseen_data = {
        "History-informed": q1_series(
            q1,
            experiment="unseen_main",
            regime="unseen_user_test",
            mode="history_informed",
        ),
        "Zero-history-trained": independent_series(
            independent, protocol="unseen_user", evaluation="internal_test"
        ),
        "GRU": q1_series(
            q1,
            experiment="unseen_gru",
            regime="unseen_user_test",
            mode="not_applicable",
        ),
        "EWMA": fixed_series(
            extract_metric(naive, regime="unseen_user_test", model="ewma")
        ),
        "Persistence": fixed_series(
            extract_metric(naive, regime="unseen_user_test", model="persistence")
        ),
    }
    external_data = {
        "History-masked": q1_series(
            q1,
            experiment="unseen_main",
            regime="goldencheetah_frozen_external",
            mode="zero_history",
        ),
        "Zero-history-trained": independent_series(
            independent, protocol="unseen_user", evaluation="frozen_external"
        ),
        "GRU": q1_series(
            q1,
            experiment="unseen_gru",
            regime="goldencheetah_frozen_external",
            mode="not_applicable",
        ),
        "EWMA": fixed_series(
            extract_metric(
                naive, regime="goldencheetah_frozen_external", model="ewma"
            )
        ),
        "Persistence": fixed_series(
            extract_metric(
                naive, regime="goldencheetah_frozen_external", model="persistence"
            )
        ),
    }

    source_rows: list[dict] = []
    for regime, payload in [
        ("temporal", temporal_data),
        ("unseen_user", unseen_data),
        ("external", external_data),
    ]:
        for method, summary in payload.items():
            seeds = (
                "deterministic"
                if method in {"EWMA", "Persistence"}
                else ("3" if method == "GRU" else "5")
            )
            source_rows.extend(series_source_rows(regime, method, summary, seeds))
    SOURCE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(
        SOURCE / "Figure_2_point_forecast_source.csv", index=False
    )

    keep = pd.concat(
        [
            effects[
                (effects.protocol == "strict_temporal")
                & (effects.evaluation == "internal_test")
                & (effects.contrast == "mixed_history_minus_independent_zero")
            ],
            effects[
                (effects.protocol == "unseen_user")
                & (effects.evaluation == "internal_test")
                & (effects.contrast == "mixed_history_minus_independent_zero")
            ],
            effects[
                (effects.protocol == "unseen_user")
                & (effects.evaluation == "frozen_external")
                & (effects.contrast == "mixed_zero_minus_independent_zero")
            ],
        ],
        ignore_index=True,
    ).rename(
        columns={
            "estimate_bpm": "delta_mae_bpm",
            "percentile_95_ci_low_bpm": "ci_low_bpm",
            "percentile_95_ci_high_bpm": "ci_high_bpm",
        }
    )
    keep["group"] = np.repeat(
        [
            "Temporal: history-informed - zero-history-trained",
            "Unseen user: history-informed - zero-history-trained",
            "Cross-source: history-masked - zero-history-trained",
        ],
        3,
    )
    keep.to_csv(SOURCE / "Figure_2_paired_effect_source.csv", index=False)
    labels = [
        f"{group} · {label}"
        for group, label in zip(
            keep.group,
            keep.horizon_seconds.map({60: "1 min", 180: "3 min", 300: "5 min"}),
        )
    ]
    effect_colors = (
        [COLORS["temporal"]] * 3
        + [COLORS["unseen"]] * 3
        + [COLORS["external"]] * 3
    )

    fig = plt.figure(figsize=(7.2, 5.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    line_panel(axes[0], temporal_data, "Strict temporal test", "a")
    line_panel(axes[1], unseen_data, "Unseen-user test", "b")
    line_panel(axes[2], external_data, "Frozen cross-source test", "c")
    for ax in axes:
        ax.set_ylim(5, 13)
    handles: list = []
    names: list[str] = []
    for ax in axes:
        ax_handles, ax_names = ax.get_legend_handles_labels()
        for handle, name in zip(ax_handles, ax_names):
            if name not in names:
                handles.append(handle)
                names.append(name)
    fig.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        fontsize=5.6,
    )
    ax_forest = fig.add_subplot(gs[1, :])
    forest(
        ax_forest,
        keep,
        labels,
        effect_colors,
        "d",
        "Paired ΔMAE (bpm); negative favours first strategy",
    )
    ax_forest.set_xlim(-0.22, 0.08)
    save(fig, "Figure_2_primary_performance")


def main() -> None:
    style()
    SOURCE.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2_multiseed()
    figure3_multiseed()
    figure4_multiseed()
    figure4_multiseed(
        reporting_threshold=30, output_stem="Figure_3_sport_shift_PMEA"
    )
    supplementary_figure1_multiseed()
    supplementary_figure1_multiseed(
        include_gender=False,
        output_stem="Supplementary_Figure_1_ablation_sensitivity_PMEA",
    )
    graphical_abstract()
    print(f"Generated figures in {FIGURES}")


if __name__ == "__main__":
    main()
