import os
import pandas as pd
import matplotlib.pyplot as plt


def load_all_metrics(parent_dir):
    data = []

    # 🔒 Specific list of labels (folders)
    label_folders = [
        "Overflow-Underflow",
        "Re-entrancy",
        "Timestamp-Dependency",
        "TOD",
        "tx.origin",
        "Unchecked-Send",
        "Unhandled-Exceptions"
    ]

    for label_folder in label_folders:
        label_path = os.path.join(
            parent_dir, label_folder, "global_metrics_report_0.1.csv")
        if os.path.isfile(label_path):
            df = pd.read_csv(label_path)
            df["Label"] = label_folder
            data.append(df)
        else:
            print(f"⚠️ Warning: Missing file in {label_folder}")

    return pd.concat(data, ignore_index=True)


def plot_metric_comparison(df, metric, save_dir):
    plt.figure(figsize=(12, 6))

    labels = sorted(df["Label"].unique())
    x = range(len(labels))
    width = 0.25

    for idx, r in enumerate([0, 1, 2]):
        subset = df[df["Range"] == r]
        values = [
            subset[subset["Label"] == label][metric].values[0]
            if not subset[subset["Label"] == label].empty else 0
            for label in labels
        ]
        plt.bar([p + width * idx for p in x], values,
                width=width, label=f"Range {r}")

    plt.xticks([p + width for p in x], labels, rotation=30, ha='right')
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison by Label (Range 0/1/2)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(
        save_dir, f"{metric.replace(' ', '_')}_comparison.png"))
    plt.close()


if __name__ == "__main__":
    parent_dir = "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts"
    output_dir = os.path.join(parent_dir, "graphs_0.1")

    combined_df = load_all_metrics(parent_dir)

    metrics = [
        "F1 Score",
        "Precision",
        "Recall",
        "True Positives",
        "False Positives",
        "False Negatives"
    ]

    for metric in metrics:
        plot_metric_comparison(combined_df, metric, output_dir)

    print(f"✅ All metric graphs saved to: {output_dir}")
