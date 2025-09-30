
import os
import json
import pandas as pd
from openai import OpenAI

# -------- CONFIGURATION -------- #
API_KEY = "your-openai-or-free-proxy-api-key"
CONTRACT_DIR = "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts/Timestamp-Dependency"
CSV_DIR = "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts/Timestamp-Dependency"
MODEL = "gpt-3.5-turbo"
TOLERANCE = 2
# -------------------------------- #

client = OpenAI(api_key=API_KEY)

def detect_vulnerabilities_with_llm(contract_code):
    base_prompt = (
        "You are a smart contract security expert. "
        "Read the following Solidity contract and return all lines (with line numbers) "
        "where there is a Timestamp Dependency vulnerability. "
        "Only return a JSON list of line numbers and a short explanation for each, like: "
        "[{'line': 42, 'reason': 'uses block.timestamp in conditional logic'}]"
    )

    messages = [
        {"role": "system", "content": "You analyze smart contract vulnerabilities."},
        {"role": "user", "content": base_prompt + "\n\n" + contract_code}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=1000
    )

    content = response.choices[0].message.content
    try:
        result = json.loads(content)
        return result
    except Exception as e:
        print("Failed to parse response:", content)
        print("Error:", str(e))
        return []

def evaluate_detection(contract_name, detected_lines, csv_path):
    df = pd.read_csv(csv_path)
    actual_lines = set(df[df['bug type'] == 'Timestamp-Dependency']['loc'])

    matched_lines = set()
    for d in detected_lines:
        for a in actual_lines:
            if abs(d['line'] - a) <= TOLERANCE:
                matched_lines.add(a)

    false_negatives = actual_lines - matched_lines
    return {
        "contract": contract_name,
        "total": len(actual_lines),
        "detected": len(matched_lines),
        "false_negatives": len(false_negatives)
    }

def main():
    summary = []
    for file in os.listdir(CONTRACT_DIR):
        if not file.endswith(".sol"):
            continue

        contract_path = os.path.join(CONTRACT_DIR, file)
        index_part = file.split('_')[1].split('.')[0]
        csv_path = os.path.join(CSV_DIR, f"BugLog_{index_part}.csv")

        with open(contract_path, 'r', encoding='utf-8') as f:
            contract_code = f.read()

        print(f"Analyzing {file}...")
        detections = detect_vulnerabilities_with_llm(contract_code)
        result = evaluate_detection(file, detections, csv_path)
        summary.append(result)

    print("\n=== Detection Summary ===")
    for res in summary:
        print(res)

if __name__ == "__main__":
    main()
