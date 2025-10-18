from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llama_cpp import Llama
from tqdm import tqdm
import time

CHROMA_PATH = "database/chroma_uu_db_indo"

llm = Llama(
    model_path="/home/ubuntu/projek_chatbot_galang/training_model/model/taxbot_v8.gguf",
    n_ctx=4096,
    verbose=False,
    device = "cpu"
)

def format_llama_cpp_chat(messages):
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text

def build_context_from_db(db, query, top_k=5):
    """
    Ambil dokumen relevan dari Chroma DB + hitung skor rata-rata
    """
    results = db.similarity_search_with_relevance_scores(query, k=top_k)
    if len(results) == 0 or results[0][1] < 0.3:
        return "Maaf, tidak ada data yang relevan ditemukan.", "none"

    context_texts = []
    scores = []
    for doc, score in results:
        scores.append(score)
        meta = doc.metadata
        context_texts.append(
            f"UU: {meta.get('uu', '')}\n"
            f"BAB: {meta.get('bab', '')}\n"
            f"{meta.get('pasal', '')}\n"
            f"Ayat: {meta.get('ayat', '')}\n"
            f"Sumber: {meta.get('sumber', '')}"
            f"{doc.page_content}\n"
            f"Relevance Score: {score:.2f}"
        )

    max_score = max(scores)
    
    if max_score >= 0.5:
        answer_type = "specific" 
    elif max_score >= 0.3 and max_score < 0.5:
        answer_type = "complex"
    else:
        answer_type = "none"

    return "\n\n---\n\n".join(context_texts), answer_type

def infer(question, context, answer_type="specific"):
    """
    Jalankan LLaMA dengan prompt sesuai tipe jawaban
    """
    if answer_type == "none":
        return "Maaf, saya tidak memiliki pemahaman tentang hal itu."
    
    SYSTEM_TEMPLATE = f"""
    Kamu adalah asisten ahli pajak Indonesia.
    Jawaban harus faktual dan to the point dan gunakan bahasa formal.

    Tipe Jawaban: {answer_type}.
    if (tipe jawaban = specific):
    Sertakan sumber pasal dari context di akhir kalimat dengan cara yang natural, misal: "sesuai dengan Pasal .. UU Nomor .. Tahun ....".
    Sertakan sumber hukum dari context dengan format Source: Pasal {{pasal}} Ayat {{ayat}} UU {{uu}}.

    else if (tipe jawaban = complex):
    FORMAT JAWABAN AKHIR:

    Sources Used:
    {{sumber}}
    [Daftar sumber UU dari context yang digunakan (minimal 2)]

    Summary:
    [Rangkuman inti analisis]

    PILIH SATU BAGIAN SAJA di bawah ini, lalu isi dengan teks yang relevan:

    [[ Conclusion ]]
    [Tulis kesimpulan, JIKA analisis berfokus pada ringkasan temuan dan implikasi logis dari data yang ada.]

    ATAU

    [[ Recommendation ]]
    [Tulis rekomendasi, JIKA analisis berfokus pada usulan aksi, kebijakan, atau langkah perbaikan di masa depan.]

    Answer the question based only on the following context:
    {context}
    """

    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": question}
    ]
    prompt = format_llama_cpp_chat(messages)

    out = llm(
        prompt,
        max_tokens=500,
        temperature=0.2,
        top_p=0.8,
        repeat_penalty=1.2,
    )

    text = out["choices"][0]["text"].strip()
    return text

def main(question):
    start_total = time.time()
    query_text = question

    embedding_function = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    context, answer_type = build_context_from_db(db, query_text, top_k=5)

    print(f"=== Context Retrieved (type: {answer_type}) ===")
    print(context[:500] + "...\n")

    response = infer(question, context, answer_type)
    print("=== Chatbot Response ===")
    print(response)
    end_total = time.time()
    print("Total Runtime :", f"{end_total - start_total:.2f} detik")
    llm.close()

if __name__ == "__main__":
    main("Apa yang dimaksud dengan Pajak?")
