from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re

CHROMA_PATH = "chroma_uu_db_v2"

from llama_cpp import Llama

# Load model GGUF
llm = Llama(
    model_path="/home/ubuntu/projek_chatbot_galang/training_model/model/taxbot_v7.gguf",
    #n_gpu_layers=50,        
    n_ctx=4096,            
    verbose=False
)

def extract_keywords(question):
    # Hilangkan kata-kata umum (stopwords sederhana)
    stopwords = {"apa", "bagaimana", "dengan", "atas", "dan", "terhadap", "dari", "di", "yang"}
    words = re.findall(r'\b\w+\b', question.lower())
    keywords = [w for w in words if w not in stopwords]
    return keywords

def is_context_specific(context, question, threshold=0.5):
    # Cek Pasal/Ayat dulu
    import re
    if re.search(r'Pasal \d+.*Ayat \d+', context, re.IGNORECASE):
        return True

    # Cek keyword matching
    keywords = extract_keywords(question)
    matches = sum(1 for kw in keywords if kw.lower() in context.lower())
    if matches / len(keywords) >= threshold:
        return True

    return False
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
    Ambil top_k dokumen relevan dari Chroma DB
    dan gabungkan page_content + metadata sebagai context
    """
    results = db.similarity_search_with_relevance_scores(query, k=top_k)
    if not results:
        return "Maaf, tidak ada data yang relevan ditemukan."

    context_texts = []
    for doc, score in results:
        meta = doc.metadata
        context_texts.append(
            f"UU: {meta.get('uu', '')}\n"
            f"BAB: {meta.get('bab', '')}\n"
            f"{meta.get('pasal', '')}\n"
            f"Ayat: {meta.get('ayat', '')}\n"
            f"Sumber: {meta.get('sumber', '')}\n\n"
            f"Isi dan Penjelasan:\n{doc.page_content}\n"
            f"Relevance Score: {score:.2f}"
        )
    return "\n\n---\n\n".join(context_texts)

def infer(question, context):
    """
    Jalankan LLaMA dengan prompt RAG
    """
    messages = [
         {"role": "system", "content": (
            f"Answer the question based only on the following context:\n{context}\n"
            "Kamu adalah asisten ahli pajak Indonesia. "
            "Jawaban harus faktual, hanya berdasarkan context yang diberikan, "
            "sertakan sumber hukum dengan format: Pasal {pasal} Ayat {ayat} UU {uu}. "
            "Jika informasi tidak ada, jawab: 'Maaf, saya tidak memiliki pemahaman tentang hal itu.'"
        )},
        {"role": "user", "content": question}
    ]
    prompt = format_llama_cpp_chat(messages)

    out = llm(
        prompt,
        max_tokens=300,
        temperature=0.2,
        top_p=0.8,
        repeat_penalty=1.2,
    )
    text = out["choices"][0]["text"].strip()
    return text

def main():
    query_text = "Apa yang dimaksud dengan Pajak?"
    
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    context = build_context_from_db(db, query_text, top_k=5)

    print("=== Context Retrieved ===")
    print(context[:1000] + "...\n") 

    response = infer(query_text, context)
    print("=== LLaMA Response ===")
    print(response)
    llm.close()

if __name__ == "__main__":
    main()
