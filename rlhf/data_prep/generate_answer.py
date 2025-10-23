import json
import time
from tqdm import tqdm
import random
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llama_cpp import Llama
from ollama import Client
import os

# ===========================
# KONFIGURASI PATH & MODEL
# ===========================
CHROMA_PATH = "/home/ubuntu/projek_chatbot_galang/chatbot/database/chroma_uu_db_indo_v2"
INPUT_FILE = "dataset/selected_questions_v3.jsonl"
OUTPUT_FILE = "dataset/generated_responses_v3.jsonl"
CHECKPOINT_FILE = "dataset/checkpoint_responses.jsonl"

llm_local = Llama(
    model_path="/home/ubuntu/projek_chatbot_galang/chatbot/model/taxbot_v9_dpo_v1.gguf",
    n_ctx=2048,
    verbose=False,
    device="cuda"
)

#ollama_client = Client()

# ===========================
# FORMAT CHAT UNTUK LLAMA_CPP
# ===========================
def format_llama_cpp_chat(messages):
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


# ===========================
# BUILD CONTEXT DENGAN STRUKTUR JELAS
# ===========================
def build_context_from_db(db, query, top_k=3):
    results = db.similarity_search_with_score(query, k=top_k)
    if len(results) == 0 or results[0][1] > 13:
        return "Maaf, tidak ada data relevan.", ["none"], "none"

    structured_contexts = []
    source_list = []

    for i, (doc, score) in enumerate(results):
        meta = doc.metadata
        uu = meta.get("uu", "")
        pasal = meta.get("pasal", "")
        ayat = meta.get("ayat", "")
        sumber = meta.get("sumber", "")
        structured_contexts.append(
            f"""
Konteks {i+1}:
{uu} {pasal} Ayat {ayat}
(Sumber: {sumber}, Skor: {score:.2f})

Isi dan/atau Penjelasan:
{doc.page_content.strip()}
            """
        )
        source_list.append(f"UU {uu} Pasal {pasal} Ayat {ayat}")

    return structured_contexts, source_list, "true"


# ===========================
# PROMPT TEMPLATE RAG
# ===========================
def build_system_template(context, source):
    return f"""Jawab pertanyaan berdasarkan konteks berikut:
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


# ===========================
# INFERENSI DENGAN LLAMA_CPP LOKAL
# ===========================
def infer_local(question, context, source):
    SYSTEM_TEMPLATE = build_system_template(context, source)
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": question}
    ]
    prompt = format_llama_cpp_chat(messages)

    out = llm_local(
        prompt,
        max_tokens=500,
        temperature=0.3,
        top_p=0.8,
        repeat_penalty=1.1
    )

    return out["choices"][0]["text"].strip()


# ===========================
# INFERENSI DENGAN OLLAMA API
# ===========================
# def infer_ollama(question, context, source):
#     SYSTEM_TEMPLATE = build_system_template(context, source)
#     response = ollama_client.chat(
#         model="llama3.1:8b",
#         messages=[
#             {"role": "system", "content": SYSTEM_TEMPLATE},
#             {"role": "user", "content": question}
#         ]
#     )
#     return response["message"]["content"].strip()


# ===========================
# UTILITAS
# ===========================
def save_checkpoint(data, filename):
    with open(filename, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_existing_questions(filename):
    """Cegah duplikasi: ambil daftar pertanyaan yang sudah di-generate"""
    if not os.path.exists(filename):
        return set()
    with open(filename, "r", encoding="utf-8") as f:
        return {json.loads(line)["question"] for line in f}


# ===========================
# MAIN PIPELINE
# ===========================
def main():
    print("🔍 Memuat database embedding dan pertanyaan...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="/home/ubuntu/projek_chatbot_galang/chatbot/model/all-indo-e5-small-v4-matryoshka-v2",
        model_kwargs={"device": "cuda"}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    done_questions = load_existing_questions(OUTPUT_FILE)
    print(f"📚 Total pertanyaan: {len(questions)} | Sudah dikerjakan: {len(done_questions)}")

    for idx, q in enumerate(tqdm(questions, desc="Generating answers")):
        question = q["prompt"]
        if question in done_questions:
            continue

        # Ambil konteks dari database
        context, source, _ = build_context_from_db(db, question)
        context_combined = "\n\n==============================\n\n".join(context)
        source_combined = "; ".join(source)

        question_outputs = []

        # 🔹 Generate dari Llama lokal
        for i in range(5):
            try:
                ans = infer_local(question, context_combined, source_combined)
                question_outputs.append({
                    "context": context_combined,
                    "source": source_combined,
                    "question": question,
                    "answer": ans,
                    "model": "taxbot_v9"
                })
            except Exception as e:
                print(f"[⚠️ error local-{i}] {e}")

        # 🔸 Generate dari Ollama
        # for j in range(2):
        #     try:
        #         ans = infer_ollama(question, context_combined, source_combined)
        #         question_outputs.append({
        #             "context": context_combined,
        #             "source": source_combined,
        #             "question": question,
        #             "answer": ans,
        #             "model": "llama3.1:8b"
        #         })
        #     except Exception as e:
        #         print(f"[⚠️ error ollama-{j}] {e}") 

        if question_outputs:
            save_checkpoint(question_outputs, OUTPUT_FILE)
            print(f"💾 Checkpoint disimpan untuk pertanyaan ke-{idx+1}: {question[:60]}...")

        time.sleep(1.5)

    print("\n✅ Semua pertanyaan selesai digenerate!")


# ===========================
# ENTRY POINT
# ===========================
if __name__ == "__main__":
    main()
    llm_local.close()
