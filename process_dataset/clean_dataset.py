# import os
# import json

# folder_path = "/home/ubuntu/projek_chatbot_galang/process_dataset/dataset/uu_per_ayat"
# output_file = "/home/ubuntu/projek_chatbot_galang/process_dataset/dataset/dataset_uu_gabung.json"

# all_data = []

# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, "r", encoding="utf-8") as f:
#             try:
#                 data = json.load(f)
#                 for item in data:
#                     if "Isi" in item:
#                         item["Isi"] = item["Isi"].replace("\n", " ")
#                     if "Penjelasan" in item:
#                         item["Penjelasan"] = item["Penjelasan"].replace("\n", " ")
#                 all_data.extend(data)
#             except Exception as e:
#                 print(f"Error membaca {filename}: {e}")

# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(all_data, f, indent=2, ensure_ascii=False)

# print(f"Dataset berhasil digabung dan disimpan di {output_file}")


import json
import re

# Path file input & output
input_path = "dataset/dataset_uu_gabung.json"
output_path = "dataset/dataset_uu_gabung_clean.json"

# Fungsi bantu untuk membersihkan teks
def clean_data(entry):
    cleaned = entry.copy()
    
    # 1. Hapus kata "Pasal " di field "Pasal"
    if "Pasal" in cleaned and isinstance(cleaned["Pasal"], str):
        cleaned["Pasal"] = cleaned["Pasal"].replace("Pasal ", "").strip()
    
    # 2. Hapus "(Sumber : ...)" dan tanda kurung di field "Sumber"
    if "Sumber" in cleaned and isinstance(cleaned["Sumber"], str):
        cleaned["Sumber"] = re.sub(r"^\(Sumber\s*:\s*(.*?)\)$", r"\1", cleaned["Sumber"]).strip()
    
    # 3. Hapus referensi "(Sumber : ...)" di dalam field "Isi"
    if "Isi" in cleaned and isinstance(cleaned["Isi"], str):
        cleaned["Isi"] = re.sub(r"\s*\(Sumber\s*:[^)]+\)", "", cleaned["Isi"]).strip()
    
    return cleaned

# Baca file JSON asli
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Looping untuk membersihkan setiap item
cleaned_data = [clean_data(item) for item in data]

# Simpan hasil bersih ke file baru
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"✅ Data berhasil dibersihkan dan disimpan ke: {output_path}")
