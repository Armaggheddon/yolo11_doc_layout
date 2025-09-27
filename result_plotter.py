from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

PLOTS_ROOT = Path(__file__).parent / "plots"
PLOTS_ROOT.mkdir(exist_ok=True)

LABELS = ["all", "caption", "footnote", "formula", "list-item", "page-footer", "page-header", "picture", "section-header", "table", "text", "title"]
LABEL2ID = {k: v for v, k in enumerate(LABELS)}

@dataclass
class Metrics:
    model_size: str
    train_iter: int
    box_precision: list[float]
    recall: list[float]
    map50: list[float]
    map50_95: list[float]

    def __getitem__(self, key):
        """ Return the metrics for a given label key."""
        if key in LABEL2ID:
            idx = LABEL2ID[key]
            return [
                self.box_precision[idx],
                self.recall[idx],
                self.map50[idx],
                self.map50_95[idx],
            ]
    
    def get_metric(self, metric_name: str):
        """ Return the metric values for all labels."""
        if metric_name == "box_precision":
            return self.box_precision
        elif metric_name == "recall":
            return self.recall
        elif metric_name == "map50":
            return self.map50
        elif metric_name == "map50_95":
            return self.map50_95
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        
    @staticmethod
    def available_metrics():
        return ["box_precision", "recall", "map50", "map50_95"]
    
train_1 = Metrics(
    "yolo11n",
    1,
    [0.83, 0.903, 0.703, 0.87, 0.807, 0.911, 0.885, 0.793, 0.849, 0.824, 0.896, 0.691],
    [0.768, 0.794, 0.487, 0.704, 0.854, 0.878, 0.645, 0.851, 0.838, 0.841, 0.873, 0.682],
    [0.85, 0.903, 0.595, 0.815, 0.891, 0.919, 0.869, 0.901, 0.909, 0.883, 0.937, 0.723],
    [0.636, 0.762, 0.418, 0.563, 0.726, 0.514, 0.515, 0.804, 0.551, 0.805, 0.762, 0.576]
)

train_2 = Metrics(
    "yolo11n",
    2,
    [0.884, 0.905, 0.852, 0.883, 0.892, 0.88, 0.936, 0.852, 0.918, 0.828, 0.91, 0.869],
    [0.84, 0.853, 0.664, 0.81, 0.878, 0.933, 0.859, 0.85, 0.878, 0.835, 0.901, 0.779],
    [0.906, 0.929, 0.776, 0.878, 0.922, 0.958, 0.958, 0.907, 0.948, 0.874, 0.951, 0.865],
    [0.719, 0.814, 0.552, 0.649, 0.785, 0.641, 0.714, 0.792, 0.692, 0.699, 0.811, 0.757]
)

train_3 = Metrics(
    "yolo11n",
    3,
    [0.871, 0.902, 0.81, 0.871, 0.886, 0.858, 0.942, 0.85, 0.909, 0.824, 0.896, 0.831],
    [0.819, 0.842, 0.58, 0.778, 0.844, 0.932, 0.834, 0.852, 0.869, 0.825, 0.893, 0.758],
    [0.892, 0.922, 0.719, 0.856, 0.911, 0.948, 0.951, 0.91, 0.941, 0.872, 0.944, 0.839],
    [0.694, 0.805, 0.494, 0.617, 0.759, 0.62, 0.687, 0.786, 0.659, 0.686, 0.791, 0.728]
)

train_4 = Metrics(
    "yolo11n",
    4,
    [0.898, 0.93, 0.853, 0.879, 0.909, 0.911, 0.939, 0.874, 0.912, 0.851, 0.918, 0.904],
    [0.847, 0.862, 0.728, 0.796, 0.891, 0.938, 0.837, 0.875, 0.882, 0.835, 0.91, 0.769],
    [0.917, 0.945, 0.805, 0.881, 0.93, 0.968, 0.954, 0.93, 0.948, 0.893, 0.955, 0.88],
    [0.716, 0.838, 0.551, 0.667, 0.785, 0.629, 0.66, 0.82, 0.64, 0.716, 0.798, 0.777]
)

train_5 = Metrics(
    "yolo11s",
    5,
    [0.908, 0.927, 0.878, 0.891, 0.918, 0.915, 0.938, 0.881, 0.931, 0.866, 0.929, 0.909],
    [0.862, 0.876, 0.715, 0.848, 0.898, 0.947, 0.859, 0.88, 0.895, 0.859, 0.919, 0.786],
    [0.929, 0.956, 0.827, 0.905, 0.941, 0.971, 0.959, 0.938, 0.954, 0.905, 0.962, 0.905],
    [0.754, 0.86, 0.586, 0.711, 0.817, 0.685, 0.716, 0.834, 0.689, 0.747, 0.824, 0.824]
)

train_6 = Metrics(
    "yolo11m",
    6,
    [0.918, 0.946, 0.936, 0.867, 0.929, 0.911, 0.955, 0.913, 0.937, 0.875, 0.947, 0.883],
    [0.862, 0.867, 0.746, 0.836, 0.9, 0.952, 0.843, 0.873, 0.893, 0.86, 0.916, 0.799],
    [0.933, 0.954, 0.862, 0.899, 0.944, 0.972, 0.958, 0.938, 0.956, 0.907, 0.965, 0.902],
    [0.766, 0.865, 0.592, 0.72, 0.817, 0.684, 0.768, 0.84, 0.722, 0.753, 0.843, 0.824]
)

train_7 = Metrics(
    "yolo11n",
    7,
    [0.865, 0.893, 0.833, 0.832, 0.881, 0.899, 0.914, 0.834, 0.89, 0.769, 0.908, 0.866],
    [0.842, 0.863, 0.67, 0.81, 0.894, 0.92, 0.833, 0.866, 0.88, 0.87, 0.884, 0.775],
    [0.902, 0.941, 0.756, 0.857, 0.926, 0.946, 0.949, 0.925, 0.941, 0.866, 0.947, 0.868],
    [0.723, 0.83, 0.513, 0.642, 0.799, 0.648, 0.752, 0.809, 0.698, 0.694, 0.811, 0.754]
)

train_8 = Metrics(
    "yolo11n",
    8,
    [0.889, 0.913, 0.839, 0.866, 0.901, 0.887, 0.921, 0.86, 0.913, 0.856, 0.916, 0.905],
    [0.836, 0.852, 0.67, 0.799, 0.878, 0.932, 0.844, 0.854, 0.876, 0.829, 0.901, 0.761],
    [0.906, 0.936, 0.754, 0.874, 0.925, 0.959, 0.941, 0.922, 0.944, 0.889, 0.953, 0.868],
    [0.715, 0.825, 0.513, 0.651, 0.791, 0.638, 0.696, 0.807, 0.672, 0.708, 0.804, 0.757]
)

train_9 = Metrics(
    "yolo11n",
    9,
    [0.889, 0.927, 0.855, 0.876, 0.9, 0.915, 0.943, 0.894, 0.899, 0.83, 0.916, 0.828],
    [0.852, 0.873, 0.698, 0.833, 0.89, 0.933, 0.844, 0.871, 0.888, 0.845, 0.908, 0.793],
    [0.915, 0.946, 0.786, 0.886, 0.929, 0.96, 0.954, 0.93, 0.949, 0.885, 0.955, 0.88],
    [0.716, 0.838, 0.54, 0.669, 0.787, 0.636, 0.666, 0.821, 0.637, 0.714, 0.798, 0.773]
)

def plot_xy_metric(
        destination: Path,
        metric_name: str,
        models_metric: list[Metrics],
        highlight: Metrics = None
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for model_results in models_metric:
        plt.plot(
            LABELS,
            model_results.get_metric(metric_name),
            marker='o',
            label=f"{model_results.model_size}:{model_results.train_iter}",
            alpha=0.7 if highlight else 1.0
        )
    if highlight:
        plt.plot(
            LABELS,
            highlight.get_metric(metric_name),
            marker='o',
            label=f"Highlighted {highlight.model_size}:{highlight.train_iter}",
            color='red',
            linewidth=2.5
        )

    plt.xlabel('Labels')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} per Label')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(destination / f"{metric_name}_per_label.png")
    plt.close()


def plot_metric_bars_percentage_improvement(
    destination: Path,
    metric_name: str,
    models_metric: list[Metrics],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    num_labels = len(LABELS)
    num_models = len(models_metric)
    x = np.arange(num_labels)
    width = 0.8 / num_models  # Adjust bar width based on number of models

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Get all metric values: list of lists, one inner list per model
    all_metrics = [model.get_metric(metric_name) for model in models_metric]

    # Find the worst performer (baseline) for each label
    baselines = np.min(all_metrics, axis=0)
    baseline_indices = np.argmin(all_metrics, axis=0)

    for i, model_results in enumerate(models_metric):
        metric_values = np.array(all_metrics[i])
        
        # Calculate percentage improvement over the dynamic baseline for each label
        improvements = np.where(baselines > 0, (metric_values - baselines) / baselines * 100, 0)
        
        # Set improvement to a small negative value for the baseline bar to indicate it
        # This bar will not be plotted, but the space is reserved.
        improvements[baseline_indices == i] = np.nan

        bar_positions = x - (width * num_models / 2) + (i + 0.5) * width
        ax.bar(bar_positions, improvements, width, label=f"{model_results.model_size}:{model_results.train_iter}")

    plt.xlabel('Labels')
    plt.ylabel(f'Percentage Improvement in {metric_name.replace("_", " ").title()}')
    plt.title(f'Percentage Improvement vs. Worst Performer in {metric_name.replace("_", " ").title()} per Label')
    plt.xticks(x, LABELS, rotation=45, ha="right")
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # Add a note about the baseline
    plt.figtext(0.5, 0.01, 'Note: For each label, the baseline (worst performer) is not shown. All other bars show % improvement over it.', 
        ha='center', fontsize=8, style='italic')

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for figtext
    plt.savefig(destination / f"{metric_name}_percentage_improvement_per_label.png")
    plt.close()

# yolo11n metric comparisons
for m in Metrics.available_metrics():
    plot_xy_metric(
        PLOTS_ROOT / "yolo11n_scores", 
        m, 
        [train_1, train_2, train_3, train_4, train_7, train_8, train_9])
    plot_metric_bars_percentage_improvement(
        PLOTS_ROOT / "yolo11n_scores",
        m,
        [train_1, train_2, train_3, train_4, train_7, train_8, train_9]
    )

# best yolo11n performers
for m in Metrics.available_metrics():
    plot_xy_metric(
        PLOTS_ROOT / "yolo11n_best", 
        m, 
        [train_4, train_9])
    plot_metric_bars_percentage_improvement(
        PLOTS_ROOT / "yolo11n_best",
        m,
        [train_4, train_9]
    )

# plot best yolo11n with s and m
for m in Metrics.available_metrics():
    plot_xy_metric(
        PLOTS_ROOT / "n_s_m_comparison", 
        m, 
        [train_4, train_5, train_6]
    )
    plot_metric_bars_percentage_improvement(
        PLOTS_ROOT / "n_s_m_comparison",
        m,
        [train_4, train_5, train_6]
    )

    