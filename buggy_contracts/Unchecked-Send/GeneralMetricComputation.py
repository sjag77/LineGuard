import os
import pandas as pd


def group_lines(lines):
    lines = sorted(lines)
    if not lines:
        return []
    groups = []
    block = [lines[0]]
    for i in range(1, len(lines)):
        if lines[i] - block[-1] <= 1:
            block.append(lines[i])
        else:
            groups.append(block)
            block = [lines[i]]
    groups.append(block)
    return groups


def compute_global_metrics(directory):
    results = []

    for range_tol in [0, 1, 2]:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for i in range(2, 51):
            real_path = os.path.join(directory, f"BugLog_{i}.csv")
            llm_path = os.path.join(directory, f"bugLog_{i}.1_llm.csv")

            if not os.path.exists(real_path) or not os.path.exists(llm_path):
                print(f"Skipping file pair: {real_path} or {llm_path} missing")
                continue

            real_df = pd.read_csv(real_path)
            llm_df = pd.read_csv(llm_path)

            real_lines = real_df["loc"].dropna().astype(int).tolist()
            llm_lines = llm_df["loc"].dropna().astype(int).tolist()
            grouped_llm = group_lines(llm_lines)

            matched_real = set()
            for real_line in real_lines:
                for group in grouped_llm:
                    if any(abs(real_line - pred_line) <= range_tol for pred_line in group):
                        matched_real.add(real_line)
                        break

            tp = len(matched_real)
            fn = len(real_lines) - tp

            fp = 0
            for group in grouped_llm:
                if not any(
                    abs(pred_line - real_line) <= range_tol
                    for pred_line in group
                    for real_line in real_lines
                ):
                    fp += 1

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "Range": range_tol,
            "True Positives": total_tp,
            "False Positives": total_fp,
            "False Negatives": total_fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    directory = "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts/Unchecked-Send"
    output_df = compute_global_metrics(directory)
    output_path = os.path.join(directory, "global_metrics_report_0.1.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Global metrics report saved to {output_path}")
