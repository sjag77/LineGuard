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
            print(f"⚠️ Skipping file pair: {real_path} or {llm_path} missing")
            continue

        try:
            real_df = pd.read_csv(real_path)
            llm_df = pd.read_csv(llm_path)

            # Normalize column names
            real_df.columns = real_df.columns.str.strip().str.lower()
            llm_df.columns = llm_df.columns.str.strip().str.lower()

            # Check if 'loc' exists in both files
            if 'loc' not in real_df.columns:
                print(f"❌ 'loc' column missing in {real_path}")
                continue
            if 'loc' not in llm_df.columns:
                print(
                    f"⚠️ 'loc' column missing in {llm_path} — treating all as false negatives")

                false_negatives = real_df['loc'].dropna().astype(int).tolist()
            else:
                llm_lines = llm_df['loc'].dropna().astype(int).tolist()

                if not llm_lines:
                    false_negatives = real_df['loc'].dropna().astype(
                        int).tolist()
                else:
                    grouped_llm = group_lines(llm_lines)
                    false_negatives = []
                    for real_line in real_df['loc'].dropna().astype(int).tolist():
                        found = any(any(abs(line - real_line) <= 0 for line in group)
                                    for group in grouped_llm)
                        if not found:
                            false_negatives.append(real_line)

            for loc in false_negatives:
                results.append({"File Index": i, "False Negative loc": loc})

        except Exception as e:
            print(f"🚨 Error processing index {i}: {e}")
            continue

    return pd.DataFrame(results)


if __name__ == "__main__":
    directory = "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts/Timestamp-Dependency"
    output_df = detect_false_negatives(directory)

    output_path = os.path.join(directory, "false_negatives_report.csv")
    output_df.to_csv(output_path, index=False)

    print(f"✅ Report saved to {output_path}")
