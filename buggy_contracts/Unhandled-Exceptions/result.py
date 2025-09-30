import os

# Adjust this to your directory path
solidity_folder = "/Users/sj_ag77/Desktop/proposal/SolidiFI-benchmark-master/buggy_contracts/Unhandled-Exceptions"  # Update if needed

total_lines = 0
line_counts = []

for i in range(1, 51):
    filename = f"buggy_{i}.sol"
    filepath = os.path.join(solidity_folder, filename)

    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            count = sum(1 for line in lines if line.strip())  # non-empty
            total_lines += count
            line_counts.append((filename, count))
    else:
        print(f"Missing: {filename}")

# Print per-file counts (optional)
for name, count in line_counts:
    print(f"{name:<15} | {count}")

# Print total
print("\n-------------------------")
print(f"Total Non-Empty Lines: {total_lines}")
