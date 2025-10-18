import os
import json

folder_path = "/home/ubuntu/projek_chatbot_galang/process_dataset/dataset/uu_per_ayat"
output_file = "/home/ubuntu/projek_chatbot_galang/process_dataset/dataset/dataset_uu_gabung.json"

all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for item in data:
                    if "Isi" in item:
                        item["Isi"] = item["Isi"].replace("\n", " ")
                    if "Penjelasan" in item:
                        item["Penjelasan"] = item["Penjelasan"].replace("\n", " ")
                all_data.extend(data)
            except Exception as e:
                print(f"Error membaca {filename}: {e}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"Dataset berhasil digabung dan disimpan di {output_file}")
