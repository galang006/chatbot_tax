DATA_FOLDER_SPECIFIC = "/home/ubuntu/projek_chatbot_galang/process_dataset/dataset/uu_per_ayat"
DATA_FOLDER_COMPLEX = "/home/ubuntu/projek_chatbot_galang/process_dataset/dataset/uu_per_pasal"
OUTPUT_FILE = "dataset/uu_per_pasal_generated.jsonl"
MODEL_NAME = "gemma3:4b"

CHECKPOINT_FILE = "checkpoint.jsonl"
CHECKPOINT_INTERVAL = 10  

TEMP_QUESTION = 0.9
TEMP_ANSWER = 0.2

PROMPT_QUESTION_SPECIFIC = """Buat 5 pertanyaan faktual berdasarkan isi pasal berikut.
⚠️ Sangat penting: Jangan sebutkan kata "pasal", "dokumen", "UU", atau kata referensi lain yang mengacu pada teks itu sendiri. 
⚠️ Jangan gunakan frasa seperti "menurut pasal ini", "berdasarkan dokumen", atau sejenisnya.
⚠️ Pertanyaan harus terdengar natural, seperti orang awam bertanya.
⚠️ Jangan ulang kata-kata persis dari teks.
Contoh:
   - Bagus: "Siapa yang wajib membayar iuran tersebut?"
   - Buruk: "Siapa yang wajib membayar iuran menurut pasal ini?"

Tuliskan pertanyaan dalam format:
1. ...
2. ...
3. ...
4. ...
5. ...

Isi Pasal:
{isi}

Pertanyaan:"""

PROMPT_QUESTION_COMPLEX = """Buat 5 pertanyaan kompleks atau analitis berdasarkan isi pasal berikut.
Pertanyaan bisa mengaitkan implikasi hukum atau contoh kasus.
Pertanyaan harus terdengar natural, seperti orang awam bertanya.
⚠️ Jangan ulang kata yang persis ada di teks, dan jangan menyebutkan nama UU atau pasal secara literal. 

Tuliskan pertanyaan dalam format:
1. ...
2. ...
3. ...
4. ...
5. ...

Isi Pasal:
{isi}

Pertanyaan:"""

PROMPT_ANSWER_SPECIFIC = """Jawablah pertanyaan berikut berdasarkan isi pasal yang diberikan.

ATURAN:
- Jawaban harus faktual dan to the point
- Sertakan sumber pasal di akhir kalimat dengan cara yang natural, misal: "sesuai dengan Pasal .. UU Nomor .. Tahun ....".
- Gunakan bahasa formal

Isi Pasal:
{isi}

Sumber Pasal:
{sumber}

Pertanyaan: {question}

Jawaban:"""

PROMPT_ANSWER_COMPLEX = """Jawablah pertanyaan berikut berdasarkan isi pasal yang diberikan.

ATURAN:
- Jawaban berupa analisis/studi kasus.
- Gunakan bahasa formal dan akademis.

FORMAT JAWABAN AKHIR:

Sources Used:
{sumber}
[Daftar sumber yang digunakan (minimal 2)]

Summary:
[Rangkuman inti analisis]

PILIH SATU BAGIAN SAJA di bawah ini, lalu isi dengan teks yang relevan:

[[ Conclusion ]]
[Tulis kesimpulan, JIKA analisis berfokus pada ringkasan temuan dan implikasi logis dari data yang ada.]

ATAU

[[ Recommendation ]]
[Tulis rekomendasi, JIKA analisis berfokus pada usulan aksi, kebijakan, atau langkah perbaikan di masa depan.]

Isi Pasal:
{isi}

Pertanyaan: {question}

Jawaban:"""

