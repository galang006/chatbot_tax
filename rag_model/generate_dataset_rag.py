from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
import ollama
from tqdm import tqdm
import os

CHROMA_PATH = "database/chroma_uu_db_indo_v2"
OUTPUT_JSONL = "database/generated_qa_dataset_v4.jsonl"

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

def generate_questions(text, n=3):
    prompt = f"""
    Anda adalah generator pertanyaan jawaban sintetis. 
    Diberikan sebuah potongan teks yang menjelaskan suatu topik, buatlah {n} pertanyaan contoh yang bisa diajukan pengguna dan bisa dijawab langsung dari isi teks. 
    Pertanyaan harus bisa dijawab dengan beberapa kata atau kalimat singkat.

    ⚠️ Penting:
    - Jangan secara eksplisit mereferensi teks, misalnya: "dalam teks tersebut", "dalam UU tersebut","menurut undang-undang ini", "dalam undang-undang teresebut".
    - Gunakan bahasa Indonesia formal namun tetap natural.
    - Jangan sebutkan kata "pasal", "UU", "dokumen", atau referensi hukum terkait topik.
    - Jangan gunakan frasa seperti "menurut pasal ini", "UU ini", "teks tersebut", atau "berdasarkan dokumen".
    - Hindari mengulang kata-kata persis dari teks.

    Teks:
    \"\"\"{text}\"\"\"

    Tulis hasilnya dalam format daftar nomor, tanpa penjelasan tambahan:
    1. Pertanyaan pertama
    2. Pertanyaan kedua
    3. Pertanyaan ketiga
    """
    try:
        response = client.generate(model="llama3.1:8b", prompt=prompt)
        raw = response["response"].strip()

        questions = [
            q.strip(" .:-")
            for q in raw.split("\n")
            if q.strip() and "?" in q
        ]

        return questions[:n]
    except Exception as e:
        print(f"⚠️ Error generate pertanyaan: {e}")
        return []

processed = 0
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        processed = sum(1 for _ in f)
    print(f"🔁 Melanjutkan dari checkpoint ({processed} baris sudah diproses).")

for i in tqdm(range(processed, len(documents)), desc="Generating questions"):
    text = documents[i].strip()
    if len(text) < 50:
        continue
    if len(text) > 1500:
        text = text[:1500]

    questions = generate_questions(text)
    if not questions:
        continue

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        for q in questions:
            record = {
                "global_chunk_id": i,
                "text": text, 
                "questions": q}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if (i + 1) % 25 == 0:
        print(f"💾 Checkpoint tersimpan di {OUTPUT_JSONL} ({i + 1} chunk).")

print(f"\n✅ Selesai! Total {len(documents)} chunk diproses.")
print(f"📝 Dataset disimpan di: {OUTPUT_JSONL}")
