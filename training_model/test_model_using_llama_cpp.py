from llama_cpp import Llama

# Load model GGUF
llm = Llama(
    model_path="/home/ubuntu/projek_chatbot_galang/training_model/model/taxbot_v8.gguf",
    #n_gpu_layers=50,        
    n_ctx=4096,            
    verbose=False
)

def format_llama_cpp_chat(messages):
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"  
    return text

def infer(question):
    context = "Dalam Undang-Undang ini yang dimaksud dengan:\n1.\nPajak adalah kontribusi wajib kepada negara yang terutang oleh orang pribadi atau badan yang bersifat memaksa berdasarkan Undang-Undang, dengan tidak mendapatkan imbalan secara langsung dan digunakan untuk keperluan negara bagi sebesar-besarnya kemakmuran rakyat. (Sumber : UU 28 Tahun 2007, Tanggal Berlaku : 1 Januari 2008)"
    messages = [
        {"role": "system", "content": f"{context}. Kamu adalah asisten ahli pajak Indonesia dan disediakan context mengenai pertanyaan. Kamu hanya boleh menjawab pertanyaan seputar pajak. Jika pertanyaannya tidak terkait pajak dan informasi tidak ditemukan, jawab dengan: 'Maaf, saya tidak memiliki pemahaman tentang hal itu. Jawaban harus faktual dan menyebutkan sumber hukum (pasal/UU) di akhir."},
        {"role": "user", "content": f"{question}"}
    ]
    
    prompt = format_llama_cpp_chat(messages)
    
    out = llm(
        prompt,
        max_tokens=200,
        temperature=0.2,
        top_p=0.8,
        repeat_penalty=1.2,
    )
    
    text = out["choices"][0]["text"].strip()
    return text


q = "Apa sanksi atas keterlambatan pelaporan SPT Tahunan badan?"
print(infer(q))

#{"messages": [{"role": "system", "content": "Kamu adalah asisten ahli pajak Indonesia. Jawaban harus faktual dan menyebutkan sumber hukum (pasal/UU) di akhir."}, {"role": "user", "content": "Jadi, apa sih yang dimaksud dengan \"pajak\" itu sebenarnya?"}, {"role": "assistant", "content": "Pajak adalah kontribusi wajib kepada negara yang terutang oleh orang pribadi atau badan, bersifat memaksa berdasarkan Undang-Undang, dengan tidak mendapatkan imbalan secara langsung dan digunakan untuk keperluan negara bagi sebesar-besarnya kemakmuran rakyat, sesuai dengan Pasal 1 Ayat 1 UU Nomor 28 Tahun 2007.\nSource: Pasal 1 Ayat 1 UU Nomor 28 Tahun 2007"}]}