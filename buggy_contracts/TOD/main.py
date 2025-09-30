import os
import json
import pandas as pd

# Paths
data_dir = "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts/TOD"
json_file = "vulnerabilities9.json"

# Load existing JSON (or start new)
if os.path.exists(json_file):
    with open(json_file, "r") as f:
        vulnerabilities = json.load(f)
else:
    vulnerabilities = []

# Get list of buggy*.sol and BugLog_*.csv files
sol_files = sorted([f for f in os.listdir(data_dir)
                   if f.startswith("buggy_") and f.endswith(".sol")])
csv_files = sorted([f for f in os.listdir(data_dir)
                   if f.startswith("BugLog_") and f.endswith(".csv")])

# Existing contract names
existing_names = {entry["name"] for entry in vulnerabilities}

for sol_file, csv_file in zip(sol_files, csv_files):
    name = sol_file
    if name in existing_names:
        print(f"Skipping {name} (already in JSON)")
        continue

    # Extract pragma and source
    sol_path = os.path.join(data_dir, sol_file)
    pragma, source = "unknown", "none"
    with open(sol_path, "r") as f:
        for line in f.readlines()[:15]:
            line = line.strip()
            if line.startswith("pragma solidity"):
                pragma = line.split("solidity")[1].strip("; ").strip()
            if "http" in line or "https" in line:
                for part in line.split():
                    if part.startswith("http"):
                        source = part
                        break

    # Load CSV metadata
    csv_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(csv_path)
    df["bug type"] = df["bug type"].str.replace(
        "Re\\+AC0-erntrancy", "Arithmetic", regex=True)

    vulnerabilities_list = [
        {"lines": [int(row["loc"])], "category": row["bug type"]}
        for _, row in df.iterrows()
    ]

    # Append entry
    entry = {
        "name": name,
        "path": f"dataset/front_running/{name}",
        "pragma": pragma,
        "source": source,
        "vulnerabilities": vulnerabilities_list
    }
    vulnerabilities.append(entry)
    print(f"Added {name} with {len(vulnerabilities_list)} vulnerabilities")

# Save JSON
with open(json_file, "w") as f:
    json.dump(vulnerabilities, f, indent=4)
print(f"\nUpdated {json_file} with {len(vulnerabilities)} contracts.")
