import os
import pandas as pd
import matplotlib.pyplot as plt


def load_paper_false_negatives():
    return {
        "Re-entrancy": {"Oyente": 1008, "Securify": 232, "Mythril": 1085, "SmartCheck": 1343, "Manticore": 1250, "Slither": 0},
        "Timestamp dep": {"Oyente": 1381, "Securify": None, "Mythril": 810, "SmartCheck": 902, "Manticore": None, "Slither": 537},
        "Unchecked-send": {"Oyente": None, "Securify": 499, "Mythril": 389, "SmartCheck": None, "Manticore": None, "Slither": None},
        "Unhandled exp": {"Oyente": 1052, "Securify": 673, "Mythril": 756, "SmartCheck": 1325, "Manticore": None, "Slither": 457},
        "TOD": {"Oyente": 1199, "Securify": 263, "Mythril": None, "SmartCheck": None, "Manticore": None, "Slither": None},
        "Integer flow": {"Oyente": 898, "Securify": None, "Mythril": 1069, "SmartCheck": 1072, "Manticore": 1196, "Slither": None},
        "tx.origin": {"Oyente": None, "Securify": None, "Mythril": 445, "SmartCheck": 1239, "Manticore": None, "Slither": 0}
    }


def load_llm_false_negatives_by_range(parent_dir):
    fn_data = {
        0: {},
        1: {},
        2: {}
    }

    label_folders = {
        "Re-entrancy": "Re-entrancy",
        "Timestamp dep": "Timestamp-Dependency",
        "Unchecked-send": "Unchecked-Send",
        "Unhandled exp": "Unhandled-Exceptions",
        "TOD": "TOD",
        "Integer flow": "Overflow-Underflow",
        "tx.origin": "tx.origin"
    }

    for label, folder in label_folders.items():
        path = os.path.join(parent_dir, folder, "global_metrics_report_0.1.csv")
        if not os.path.isfile(path):
            print(f"⚠️ Missing file: {path}")
            for r in [0, 1, 2]:
                fn_data[r][label] = None
            continue

        try:
            df = pd.read_csv(path)
            for r in [0, 1, 2]:
                row = df[df["Range"] == r]
                if row.empty:
                    fn_data[r][label] = None
                else:
                    fn_data[r][label] = int(row["False Negatives"].values[0])
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")
            for r in [0, 1, 2]:
                fn_data[r][label] = None

    return fn_data


def plot_fn_comparison_per_range(paper_data, llm_data, save_path):
    tools = ["Oyente", "Securify", "Mythril", "SmartCheck", "Manticore", "Slither", "LLM"]
    for r in [0, 1, 2]:
        plt.figure(figsize=(14, 6))
        labels = list(paper_data.keys())
        x = range(len(labels))
        bar_width = 0.1

        for idx, tool in enumerate(tools):
            values = []
            for label in labels:
                if tool == "LLM":
                    values.append(llm_data[r].get(label, None))
                else:
                    values.append(paper_data[label].get(tool, None))

            plt.bar(
                [i + bar_width * idx for i in x],
                [v if v is not None else 0 for v in values],
                width=bar_width,
                label=tool
            )

        plt.xticks([i + bar_width * 3 for i in x], labels, rotation=30, ha="right")
        plt.ylabel("False Negatives")
        plt.title(f"False Negatives Comparison – LLM vs Tools (Range {r})")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"false_negatives_comparison_tools_LLM{r}.png"))
        plt.close()


if __name__ == "__main__":
    parent_dir = "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts"
    save_path = os.path.join(parent_dir, "graphs_0.1")

    paper_fn_data = load_paper_false_negatives()
    llm_fn_data = load_llm_false_negatives_by_range(parent_dir)

    plot_fn_comparison_per_range(paper_fn_data, llm_fn_data, save_path)

    print("✅ Saved 3 comparison charts with LLM vs Tools (range 0, 1, 2)")
