
import os
import pandas as pd


def group_lines(lines):
    lines = sorted(lines)
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


def detect_false_negatives(directory):
    results = []
    for i in range(2, 51):
        real_path = os.path.join(directory, f"BugLog_{i}.csv")
        llm_path = os.path.join(directory, f"bugLog_{i}_llm.csv")

        if not os.path.exists(real_path) or not os.path.exists(llm_path):
            print(f"Skipping file pair: {real_path} or {llm_path} missing")
            continue

        real_df = pd.read_csv(real_path)
        llm_df = pd.read_csv(llm_path)

        llm_lines = llm_df["loc"].dropna().astype(int).tolist()
        if not llm_lines:
            false_negatives = real_df["loc"].dropna().astype(int).tolist()
        else:
            grouped_llm = group_lines(llm_lines)
            false_negatives = []
            for real_line in real_df["loc"].dropna().astype(int).tolist():
                found = False
                for group in grouped_llm:
                    if any(abs(line - real_line) <= 0 for line in group):
                        found = True
                        break
                if not found:
                    false_negatives.append(real_line)

        for loc in false_negatives:
            results.append({"File Index": i, "False Negative loc": loc})

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Set this to your directory path
    directory = "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts/Unhandled-Exceptions"
    output_df = detect_false_negatives(directory)
    output_df.to_csv(
        "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts/Unhandled-Exceptions/false_negatives_report.csv", index=False)
    print("Report saved to false_negatives_report.csv")
