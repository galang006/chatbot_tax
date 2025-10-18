import json

INPUT_JSONL = "/home/ubuntu/projek_chatbot_galang/process_dataset/dataset/dataset_v4.jsonl"  
OUTPUT_JSONL = "dataset/uu_llama_dataset.jsonl"

SYSTEM_TEMPLATE = """Answer the question based only on the following context:
{{context}}
Kamu adalah asisten ahli pajak Indonesia.
Jawaban harus faktual dan to the point
Gunakan bahasa formal
Jika informasi tidak ada atau pertanyaan tidak ada hubungannya dengan pajak, jawab: 'Maaf, saya tidak memiliki pemahaman tentang hal itu.

Tipe Jawaban: {type}.
bila type = specific:
Sertakan sumber pasal di akhir kalimat dengan cara yang natural, misal: "sesuai dengan Pasal .. UU Nomor .. Tahun ....".
Sertakan sumber hukum dengan format Source: Pasal {{pasal}} Ayat {{ayat}} UU {{uu}}.

tipe = complex
FORMAT JAWABAN AKHIR:

Sources Used:
{{sumber}}
[Daftar sumber UU yang digunakan (minimal 2)]

Summary:
[Rangkuman inti analisis]

PILIH SATU BAGIAN SAJA di bawah ini, lalu isi dengan teks yang relevan:

[[ Conclusion ]]
[Tulis kesimpulan, JIKA analisis berfokus pada ringkasan temuan dan implikasi logis dari data yang ada.]

ATAU

[[ Recommendation ]]
[Tulis rekomendasi, JIKA analisis berfokus pada usulan aksi, kebijakan, atau langkah perbaikan di masa depan.]
"""

def system_content_for_dataset(type_):
    return SYSTEM_TEMPLATE.format(type=type_)

def convert_to_llama_format(input_file, output_file):
    out_list = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            role_type = data.get("type", "specific")  # specific / complex
            instruction = data.get("instruction", "")
            output = data.get("output", "")
            source = data.get("source", "")

            system_content = SYSTEM_TEMPLATE.format(type=role_type)

            if role_type == "specific":
                assistant_content = f"{output}\nSource: {source}"
            else:
                assistant_content = output

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": assistant_content}
            ]

            out_list.append({"messages": messages})

    with open(output_file, "w", encoding="utf-8") as f:
        for item in out_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Conversion done, saved to {output_file}, total records: {len(out_list)}")

if __name__ == "__main__":
    convert_to_llama_format(INPUT_JSONL, OUTPUT_JSONL)
