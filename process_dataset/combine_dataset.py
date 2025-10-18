import json
from pathlib import Path

input_folder = Path("dataset/qa_dataset")

output_file = Path("dataset_v3.jsonl")

jsonl_files = list(input_folder.glob("*.jsonl"))

print(f"Menemukan {len(jsonl_files)} file JSONL di folder {input_folder}")

all_data = []

for file_path in jsonl_files:
    print(f"  Membaca file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    all_data.append(data)
                except json.JSONDecodeError:
                    print(f"JSON invalid di file {file_path}: {line[:50]}...")

print(f"💾 Menyimpan gabungan ke: {output_file} ({len(all_data)} entries)")
with open(output_file, "w", encoding="utf-8") as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Selesai!")
