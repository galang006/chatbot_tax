from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
import ollama
from tqdm import tqdm
import os

CHROMA_PATH = "database/chroma_uu_db_indo"
OUTPUT_JSONL = "generated_qa_dataset.jsonl"

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
    Buat {n} pertanyaan singkat berdasarkan teks berikut. 
    Setiap pertanyaan harus bisa dijawab dengan isi teks, dan gunakan bahasa Indonesia yang formal namun natural.
    ⚠️ Sangat penting: Jangan sebutkan kata "pasal", "dokumen", "UU", atau kata referensi lain yang mengacu pada teks itu sendiri. 
    ⚠️ Jangan gunakan frasa seperti "menurut pasal ini", "berdasarkan dokumen", atau sejenisnya.
    ⚠️ Pertanyaan harus terdengar natural, seperti orang awam bertanya.
    ⚠️ Jangan ulang kata-kata persis dari teks.
    Contoh:
   - Bagus: "Siapa yang wajib membayar iuran tersebut?"
   - Buruk: "Siapa yang wajib membayar iuran menurut pasal ini?"

    Teks:
    \"\"\"{text}\"\"\"

    Tulis hasilnya dalam format daftar (tanpa penjelasan tambahan), contoh:
    1. Pertanyaan pertama
    2. Pertanyaan kedua
    3. Pertanyaan ketiga
    """
    try:
        response = client.generate(model="gemma3:4b", prompt=prompt)
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
            record = {"text": text, "questions": q}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if (i + 1) % 25 == 0:
        print(f"💾 Checkpoint tersimpan di {OUTPUT_JSONL} ({i + 1} chunk).")

print(f"\n✅ Selesai! Total {len(documents)} chunk diproses.")
print(f"📝 Dataset disimpan di: {OUTPUT_JSONL}")
