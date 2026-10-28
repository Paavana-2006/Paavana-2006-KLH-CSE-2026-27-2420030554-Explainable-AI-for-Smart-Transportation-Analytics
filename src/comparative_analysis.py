from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

BASELINE_FILE = RESULTS_DIR / "baseline_model_comparison.csv"
LITERATURE_FILE = RESULTS_DIR / "literature_model_comparison.csv"
HYBRID_FILE = RESULTS_DIR / "hybrid_selection_summary.csv"

OUTPUT_FILE = RESULTS_DIR / "comparative_model_performance.png"


# ============================================================
# LOAD RESULTS
# ============================================================

baseline = pd.read_csv(BASELINE_FILE)
literature = pd.read_csv(LITERATURE_FILE)
hybrids = pd.read_csv(HYBRID_FILE)


# ============================================================
# KEEP MULTIMODAL RESULTS
# ============================================================

baseline = baseline[
    baseline["Modality"] == "Multimodal"
].copy()

literature = literature[
    literature["Modality"] == "Multimodal"
].copy()


# ============================================================
# BASELINE MODELS
# ============================================================

baseline_models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost"
]

baseline = baseline[
    baseline["Model"].isin(baseline_models)
]


# ============================================================
# LITERATURE MODELS
# ============================================================

literature_models = [
    "SVM (Literature Comparator)",
    "FNN/MLP (Literature Comparator)"
]

literature = literature[
    literature["Model"].isin(literature_models)
]


# ============================================================
# COMBINE RESULTS
# ============================================================

baseline_plot = baseline[
    [
        "Model",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].copy()

literature_plot = literature[
    [
        "Model",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].copy()

hybrid_plot = hybrids[
    [
        "Model",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].copy()


# ============================================================
# DISPLAY NAMES
# ============================================================

literature_plot["Model"] = (
    literature_plot["Model"]
    .replace(
        {
            "SVM (Literature Comparator)": "SVM",
            "FNN/MLP (Literature Comparator)": "FNN/MLP"
        }
    )
)

hybrid_plot["Model"] = (
    hybrid_plot["Model"]
    .str.replace(
        "Hybrid ",
        "H",
        regex=False
    )
)


# ============================================================
# COMBINE
# ============================================================

plot_df = pd.concat(
    [
        baseline_plot,
        literature_plot,
        hybrid_plot
    ],
    ignore_index=True
)


# ============================================================
# FIND BEST BASELINE AND HYBRID
# ============================================================

best_baseline = baseline_plot.loc[
    baseline_plot["F1 Score"].idxmax()
]

best_hybrid = hybrid_plot.loc[
    hybrid_plot["F1 Score"].idxmax()
]


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\nBEST BASELINE")
print(
    best_baseline[
        ["Model", "F1 Score"]
    ].to_string()
)

print("\nBEST HYBRID")
print(
    best_hybrid[
        ["Model", "F1 Score"]
    ].to_string()
)


# ============================================================
# GRAPH
# ============================================================

models = plot_df["Model"].tolist()

x = np.arange(len(models))

width = 0.25

fig, ax = plt.subplots(
    figsize=(18, 10)
)


# Precision
bars1 = ax.bar(
    x - width,
    plot_df["Precision"],
    width,
    label="Precision"
)


# Recall
bars2 = ax.bar(
    x,
    plot_df["Recall"],
    width,
    label="Recall"
)


# F1
bars3 = ax.bar(
    x + width,
    plot_df["F1 Score"],
    width,
    label="F1-Score"
)


# ============================================================
# VALUE LABELS
# ============================================================

def add_labels(bars):

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.008,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )


add_labels(bars1)
add_labels(bars2)
add_labels(bars3)


# ============================================================
# HIGHLIGHT HYBRID AREA
# ============================================================

baseline_count = (
    len(baseline_plot)
    + len(literature_plot)
)

hybrid_count = len(hybrid_plot)

if hybrid_count > 0:

    start = baseline_count - 0.5
    end = baseline_count + hybrid_count - 0.5

    ax.axvspan(
        start,
        end,
        alpha=0.10
    )


# ============================================================
# BEST MODEL LABELS
# ============================================================

best_baseline_index = plot_df[
    "Model"
].tolist().index(
    best_baseline["Model"]
)

best_hybrid_index = plot_df[
    "Model"
].tolist().index(
    best_hybrid["Model"]
)


ax.annotate(
    f"Best Baseline\nF1 = {best_baseline['F1 Score']:.3f}",
    xy=(
        best_baseline_index + width,
        best_baseline["F1 Score"]
    ),
    xytext=(
        best_baseline_index + width,
        best_baseline["F1 Score"] + 0.10
    ),
    ha="center",
    fontsize=10,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->"
    )
)


ax.annotate(
    f"Best Hybrid\nF1 = {best_hybrid['F1 Score']:.3f}",
    xy=(
        best_hybrid_index + width,
        best_hybrid["F1 Score"]
    ),
    xytext=(
        best_hybrid_index + width,
        best_hybrid["F1 Score"] + 0.10
    ),
    ha="center",
    fontsize=10,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->"
    )
)


# ============================================================
# TITLES
# ============================================================

ax.set_title(
    "Comparative Model Performance on Smart Transportation Risk-Trend Prediction",
    fontsize=20,
    fontweight="bold",
    pad=20
)

ax.text(
    0.5,
    1.01,
    "Precision · Recall · F1-Score | Multimodal Dataset (19 Features, 143 Samples)",
    transform=ax.transAxes,
    ha="center",
    fontsize=12
)


ax.set_ylabel(
    "Score",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel(
    "Models / Approaches",
    fontsize=13,
    fontweight="bold"
)


ax.set_xticks(x)

ax.set_xticklabels(
    models,
    rotation=25,
    ha="right",
    fontsize=9
)


ax.set_ylim(
    0,
    max(
        plot_df[
            ["Precision", "Recall", "F1 Score"]
        ].max()
    ) + 0.15
)


ax.grid(
    axis="y",
    alpha=0.25
)


ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.06),
    ncol=3
)


# ============================================================
# BOTTOM INFORMATION
# ============================================================

fig.text(
    0.05,
    0.05,
    "Models Compared\n"
    "LR · DT · RF · GB · XGB · SVM · FNN/MLP · Hybrid Candidates",
    fontsize=11,
    fontweight="bold"
)

fig.text(
    0.38,
    0.05,
    "Key Insights\n"
    f"Best baseline: {best_baseline['Model']} "
    f"(F1 = {best_baseline['F1 Score']:.3f})\n"
    f"Best tested hybrid: {best_hybrid['Model']} "
    f"(F1 = {best_hybrid['F1 Score']:.3f})",
    fontsize=11
)

fig.text(
    0.72,
    0.05,
    "Notes\n"
    "• Same multimodal dataset\n"
    "• 143 samples, 19 features\n"
    "• 5-fold GroupKFold by State\n"
    "• Literature models are comparators",
    fontsize=11
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout(
    rect=[0, 0.13, 1, 0.92]
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved comparative analysis PNG:"
)

print(OUTPUT_FILE)