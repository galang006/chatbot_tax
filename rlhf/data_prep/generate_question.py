from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
import ollama
from tqdm import tqdm
import os

CHROMA_PATH = "/home/ubuntu/projek_chatbot_galang/rag_model/database/chroma_uu_db_indo_v2"
OUTPUT_JSONL = "dataset/generated_qa_dataset_v5.jsonl"

embedding_function = HuggingFaceEmbeddings(
    model_name="LazarusNLP/all-indo-e5-small-v4",
    model_kwargs={"device": "cuda"},
)

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function
)

client = ollama.Client()
data = db.get()
documents = data.get("documents", [])
print(f"📚 Jumlah total chunk: {len(documents)}")

def generate_two_questions(text):
    prompt = f"""
    Anda adalah generator pertanyaan jawaban sintetis untuk topik hukum pajak Indonesia.

    Diberikan gabungan 3 potongan teks, buatlah:
    1. Satu pertanyaan **spesifik** yang dapat dijawab langsung dari isi teks (misalnya terkait sanksi, prosedur, batas waktu, tarif, dll).
    2. Satu pertanyaan **studi kasus atau analitis**, yang lebih umum (misalnya "apa dampak", "bagaimana jika", atau "apa peran").

    ⚠️ Jika teks tidak mengandung informasi yang bisa dijadikan bahan pertanyaan (misalnya definisi umum, pengantar, atau tidak relevan),
    jawab hanya dengan:
    "Tidak ada pertanyaan yang bisa diajukan."

    Teks:
    \"\"\"{text}\"\"\"

    Format jawaban:
    Spesifik: ...
    Studi Kasus: ...
    """

    try:
        response = client.generate(model="llama3.1:8b", prompt=prompt)
        raw = response["response"].strip()

        # Skip jika model bilang tidak ada pertanyaan
        if "tidak ada pertanyaan" in raw.lower():
            return None, None

        # Ambil dua baris spesifik & studi kasus
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        spesifik, studi_kasus = None, None
        for line in lines:
            if line.lower().startswith("spesifik"):
                spesifik = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("studi"):
                studi_kasus = line.split(":", 1)[-1].strip()

        return spesifik, studi_kasus
    except Exception as e:
        print(f"⚠️ Error generate pertanyaan: {e}")
        return None, None

processed = 0
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        processed = sum(1 for _ in f)
    print(f"🔁 Melanjutkan dari checkpoint ({processed} baris sudah diproses).")

for i in tqdm(range(processed, len(documents) - 2), desc="Generating 2 questions per 3 chunks"):
    text = "\n".join(documents[i:i+3])
    if len(text) < 100:
        continue
    if len(text) > 2000:
        text = text[:2000]

    spesifik, studi_kasus = generate_two_questions(text)
    if not spesifik or not studi_kasus:
        continue

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        record = {
            "window_start_chunk": i,
            "window_end_chunk": i+2,
            "text": text,
            "question_spesifik": spesifik,
            "question_studi_kasus": studi_kasus
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if (i + 1) % 25 == 0:
        print(f"💾 Checkpoint tersimpan di {OUTPUT_JSONL} ({i + 1} window).")

print(f"\n✅ Selesai! Dataset disimpan di: {OUTPUT_JSONL}")
