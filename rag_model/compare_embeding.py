from langchain_huggingface  import HuggingFaceEmbeddings
import numpy as np

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text1 = "Setiap Wajib Pajak wajib mendaftarkan diri pada kantor pajak."
text2 = "Wajib Pajak harus mendapatkan Nomor Pokok Wajib Pajak."

vec1 = embedding_function.embed_query(text1)
vec2 = embedding_function.embed_query(text2)

cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"Cosine similarity: {cos_sim}")
