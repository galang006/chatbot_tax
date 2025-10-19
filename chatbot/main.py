from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llama_cpp import Llama
from tqdm import tqdm
import time

CHROMA_PATH = "database/chroma_uu_db_indo_v2"

llm = Llama(
    model_path="model/taxbot_v8.gguf",
    n_ctx=2048,
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
    results = db.similarity_search_with_score(query, k=top_k)
    if len(results) == 0 or results[0][1] > 13:
        return "Maaf, tidak ada data yang relevan ditemukan.", "none"

    elif results[0][1] <= 9:
        answer_type = "specific" 
    elif results[0][1] > 9 and results[0][1] <= 13:
        answer_type = "complex"
    else:
        answer_type = "none"

    context_texts = []
    source_list = []
    scores = []
    for doc, score in results:
        scores.append(score)
        meta = doc.metadata
        context_texts.append(
            f"{doc.page_content}\n"
        )
        source_texts = (
            f"{meta.get('uu', '')} "
            f"{meta.get('pasal', '')} "
            f"Ayat {meta.get('ayat', '')}\n"
            f"Sumber: {meta.get('sumber', '')}; "
            f"Relevance Score: {score:.2f}"
        )
        source_list.append(source_texts)

    return context_texts, source_list, answer_type

def infer(question, context, source,answer_type="specific"):
    """
    Jalankan LLaMA dengan prompt sesuai tipe jawaban
    """
    SYSTEM_TEMPLATE = f"""
    Jawab pertanyaan berdasarkan konteks berikut:
    {context}

    Kamu adalah asisten ahli pajak Indonesia.
    Jawaban harus faktual, to the point, dan menggunakan bahasa formal.
    Jika informasi tidak ada atau pertanyaan tidak berkaitan dengan pajak,
    jawab: "Maaf, saya tidak memiliki pemahaman tentang hal itu."

    Sumber konteks: {source}

    ------------------------------------------------------------
    Bila jawaban ditemukan dengan jelas di konteks:
    - Sertakan sumber pasal di akhir kalimat dengan cara yang natural,
    misalnya: "sesuai dengan Pasal ... UU Nomor ... Tahun ....".
    - Sertakan sumber hukum dengan format:
    Source: Pasal {{pasal}} Ayat {{ayat}} UU {{uu}}.

    ------------------------------------------------------------
    Bila jawaban tidak ditemukan dengan jelas di konteks:
    Gunakan FORMAT JAWABAN AKHIR berikut:

    Sources Used:
    {source}
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
    ------------------------------------------------------------
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
        model_name="model/all-indo-e5-small-v4-matryoshka-v2",
        model_kwargs={"device": "cpu"}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    context, source, answer_type = build_context_from_db(db, query_text, top_k=3)
    context_combined = "\n\n---\n\n".join(context)

    if answer_type != 'none':
        # print(f"=== Context Retrieved (type: {answer_type}) ===")
        # for i in range(3):
        #     print(f"--- Context {i+1} ---")
        #     print(context[i][:500].strip()) 
        #     print(f"\n📚 Source: {source[i]}\n")
        #     print("-" * 60)

        response = infer(question, context_combined, source, answer_type)
        print("=== Chatbot Response ===")
        print(response)
    
    else:
       print("Maaf, saya tidak memiliki pemahaman tentang hal itu.")

    end_total = time.time()
    print("Total Runtime :", f"{end_total - start_total:.2f} detik")

if __name__ == "__main__":
    while True:
        question = input("User: ")
        if question.lower() in ["exit", "quit"]:
            print("Bye!")
            break
        main(question)
