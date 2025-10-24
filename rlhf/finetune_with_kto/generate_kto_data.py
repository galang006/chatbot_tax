import json
import os

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

# ==== KONFIGURASI ====
file_paths = [
    "/home/ubuntu/projek_chatbot_galang/rlhf/data_prep/dataset/generated_responses_v2.jsonl",
    "/home/ubuntu/projek_chatbot_galang/rlhf/data_prep/dataset/generated_responses_v3.jsonl"
]

output_file = "labeled_responses.jsonl"
checkpoint_interval = 5  # simpan otomatis setiap N data

SYSTEM_TEMPLATE = """Jawab pertanyaan berdasarkan konteks berikut:
{context}

Kamu adalah asisten ahli pajak Indonesia.
Jawaban harus faktual, to the point, dan menggunakan bahasa formal.
Jika informasi tidak ada dikonteks atau pertanyaan tidak berkaitan dengan pajak,
jawab: "Maaf, saya tidak memiliki pemahaman tentang hal itu."

Sumber konteks: {source}

Bila jawaban ditemukan dengan jelas di konteks:
- Sertakan sumber pasal di akhir kalimat dengan cara yang natural,
  misalnya: "sesuai dengan Pasal {{pasal}} UU Nomor {{ayat}} Tahun {{tahun}}".
- Sertakan sumber hukum dengan format:
  Source: Pasal {{pasal}} Ayat {{ayat}} UU {{uu}}.

Bila jawaban tidak ditemukan dengan jelas di konteks:
Gunakan FORMAT JAWABAN AKHIR berikut:

Sources Used:
[Daftar sumber UU yang digunakan (minimal 2)]

Summary:
[Rangkuman inti analisis]

PILIH SATU BAGIAN SAJA di bawah ini, lalu isi dengan teks yang relevan:

[[ Conclusion ]]
[Tulis kesimpulan, JIKA analisis berfokus pada ringkasan temuan
dan implikasi logis dari data yang ada.]

ATAU

[[ Recommendation ]]
[Tulis rekomendasi, JIKA analisis berfokus pada usulan aksi,
kebijakan, atau langkah perbaikan di masa depan.]
"""

# ==== FUNGSI ====
def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_checkpoint(results):
    """Simpan data sementara ke file output"""
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"💾 Checkpoint disimpan ({len(results)} data) ke '{output_file}'")

def load_existing_results():
    """Muat hasil sebelumnya kalau file sudah ada"""
    if not os.path.exists(output_file):
        return []
    with open(output_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    all_data = []
    for path in file_paths:
        all_data.extend(read_jsonl(path))

    # Jika sudah ada file hasil sebelumnya, lanjutkan dari situ
    results = load_existing_results()
    start_index = len(results)
    print(f"🔁 Melanjutkan labeling dari indeks {start_index + 1}...\n")

    for i, item in enumerate(all_data[start_index:], start=start_index + 1):
        clear_screen()
        print("=" * 100)
        print(f"[{i}/{len(all_data)}]")
        print(f"Question:\n{item['question']}\n")
        print(f"Context:\n{item['context']}\n")
        print(f"Model Answer:\n{item['answer']}\n")
        print("-" * 100)

        # Input penilaian user (1 = bagus, 0 = tidak)
        while True:
            label_input = input("Apakah jawaban ini bagus? (1 = ya, 0 = tidak, s = skip): ").strip().lower()
            if label_input in ["1", "0", "s"]:
                break
            else:
                print("Masukkan hanya 1, 0, atau s (skip)!")

        if label_input == "s":
            print("⏭️ Dilewati.\n")
            continue

        label = True if label_input == "1" else False

        # Buat SYSTEM_PROMPT
        system_prompt = SYSTEM_TEMPLATE.format(
            context=item["context"],
            source=item["source"]
        )

        # Format hasil
        entry = {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Q: {item['question']}\nA:"}
            ],
            "completion": {
                "role": "assistant",
                "content": item["answer"]
            },
            "label": label
        }

        results.append(entry)

        # Simpan checkpoint setiap N data
        if len(results) % checkpoint_interval == 0:
            save_checkpoint(results)

    # Simpan akhir
    save_checkpoint(results)
    print("\n✅ Selesai! Semua data telah disimpan di file output.")

if __name__ == "__main__":
    main()
