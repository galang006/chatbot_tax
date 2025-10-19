# 📘 Projek Chatbot Galang — RAG untuk UU Perpajakan Indonesia

Proyek ini membangun chatbot berbasis **Retrieval-Augmented Generation (RAG)** untuk menjawab pertanyaan terkait **Undang-Undang Perpajakan di Indonesia**.  

Model utama yang digunakan adalah **SeaLLMs-v3-1.5B-Chat**, yang di-*fine-tune* menggunakan **synthetic dataset** hasil generate dari model **Gemma 3:4B**.  

Sistem RAG dikembangkan dengan menggunakan model embedding **LazarusNLP/all-indo-e5-small-v4**, yang juga di-*fine-tune* menggunakan **synthetic dataset** hasil generate dari model **Llama 3.1:8B**, dan seluruh proses retrieval dilakukan melalui **ChromaDB**.

---

## 📁 Struktur Folder

### **1. base_model/**
Berisi model dasar yang diunduh dari **Hugging Face** dan digunakan sebagai starting point sebelum fine-tuning.

---

### **2. process_dataset/**
Berisi kode untuk **membersihkan dataset**, **memproses data**, dan **menghasilkan synthetic dataset**.

#### Struktur:
process_dataset/
│
├── dataset/ # Dataset yang sudah diproses
│
├── clean_dataset.py # Menghapus karakter '\n' dari dataset
├── config.py # Variabel konfigurasi untuk generate dataset
├── convert_dataset_to_llama_ds.py # Mengubah dataset ke format kompatibel dengan LLaMA
├── extract_text_from_sdsn_pdf.py # Mengekstrak teks dari file PDF SDSN
└── generate_synthetic_dataset_using_llm.py # Membuat synthetic dataset menggunakan model LLaMA

---

### **3. rag_model/**
Berisi semua kode yang berkaitan dengan **RAG (Retrieval-Augmented Generation)**.

#### Struktur:
rag_model/
│
├── database/                        # Menyimpan Chroma database dan database embedding model
├── model/                           # Model hasil fine-tuning untuk RAG
│
├── compare_embedding.py              # Mengecek skor similarity antar embedding
├── create_db.py                      # Membuat ChromaDB menggunakan model LazarusNLP/all-indo-e5-small-v4
├── generate_dataset.py               # Menghasilkan dataset berbasis chunk untuk fine-tuning embedding model
├── test_rag.py                       # Menguji performa query RAG
└── view_chunk.py                     # Melihat hasil chunking data

---

### **4. training_model/**
Berisi kode dan notebook untuk **training dan fine-tuning model**.

#### Struktur:
training_model/
│
├── dataset/ # Dataset untuk training
├── model/ # Menyimpan seluruh model hasil training (format .gguf dan LoRA)
│
├── FT_Embedding_Models_on_Domain_Specific_Data.ipynb   # Notebook untuk training embedding model
├── test_model_using_llama_cpp.py                       # Testing model hasil training dengan llama_cpp
├── test_model_using_transformers.py                    # Testing model hasil training dengan library Transformers
├── training_model_using_SFTTrainer.ipynb               # Fine-tuning menggunakan SFTTrainer
└── training_model_using_Trainer.ipynb                  # Fine-tuning menggunakan Trainer

---

## Lingkungan dan Konfigurasi

- **`.venv/`**      — Virtual environment Python.  
- **`.env`**        — Berisi variabel lingkungan, seperti **Hugging Face Token**.  
- **`.gitignore`**  — File untuk mengecualikan folder dan file yang tidak perlu diunggah ke Git.

---

## Teknologi yang Digunakan

- **SeaLLMs/SeaLLMs-v3-1.5B-Chat** utk chatbot bahasa indonesia
- **LazarusNLP/all-indo-e5-small-v4** utk embending rag
- **llama3.1:8b** untuk generate systetic dataset 
- **LLaMA / Llama.cpp**
- **LangChain**
- **ChromaDB**
- **Hugging Face Transformers**
- **RAG Pipeline**
- **Python 3.10+**

---
## 📂 Sumber Dataset

Dataset diambil dari artikel DDTC:  
🔗 [10 UU Perpajakan yang Saat Ini Berlaku di Indonesia, Kamu Harus Tahu](https://news.ddtc.co.id/berita/nasional/1803971/10-uu-perpajakan-yang-saat-ini-berlaku-di-indonesia-kamu-harus-tahu)

Daftar Undang-Undang yang digunakan:
- UU PPN Konsolidasi setelah UU HPP  
- UU PPSP Konsolidasi  
- UU KUP Konsolidasi setelah UU HPP  
- UU PPh Konsolidasi setelah UU 6 Tahun 2023  
- UU Nomor 1 Tahun 2022  
- UU Nomor 10 Tahun 2020  
- UU Nomor 12 Tahun 1994  
- UU Nomor 14 Tahun 2002  
- UU Nomor 17 Tahun 2006  
- UU Nomor 39 Tahun 2007  

---

## 🔄 Alur Proyek

### 🧩 A. Preprocessing Data

1. **Ekstraksi & Pembersihan Data**
   - Data diambil dari sumber DDTC, kemudian dilakukan **preprocessing dan cleaning**.  
   - Hasil disimpan dalam format `.json` dengan struktur berikut:
     ```json
     {
       "UU": "",
       "BAB": "",
       "Pasal": "",
       "Ayat": "",
       "Isi": "",
       "Penjelasan": "",
       "Sumber": "",
       "AturanTerkait": ""
     }
     ```

2. **Generate Synthetic Dataset**
   - Menggunakan skrip `generate_synthetic_dataset_using_llm.py` dengan **model generator Gemma 3:4B**.  
   - Setiap **ayat** digenerate menjadi **5 pertanyaan** untuk fine-tuning **SeaLLMs-v3-1.5B-Chat**.  
   - Hasil disimpan sebagai:
     ```
     training_model/dataset/uu_dataset_chatbotv2.jsonl
     ```
     dengan sekitar **500 baris data** dalam format:
     ```json
     {"messages": [
       {"role": "system", "content": ""},
       {"role": "user", "content": ""},
       {"role": "assistant", "content": ""}
     ]}
     ```

3. **Fine-Tuning Model Chat**
   - Dilakukan fine-tuning menggunakan **LoRA** melalui notebook `training_model_using_SFTTrainer.ipynb`.  
   - Base model: `SeaLLMs-v3-1.5B-Chat`  
   - Dataset: hasil generate Gemma 3:4B  
   - Hasil model:
     ```
     lora-taxbot-SeaLLMs-v3-1.5B-Chat-v8
     ```
   - Setelah merge dengan base model, dihasilkan model final:
     ```
     merged-taxbot-SeaLLMs-v3-1.5B-Chat-v8
     ```

4. **Konversi Model ke Format GGUF**
   - Model diubah ke format **GGUF** menggunakan **Llama.cpp** agar lebih ringan dan efisien.  
   - Hasil:
     ```
     /projek_chatbot_galang/rag_model/model/taxbot_v8.gguf
     ```

---

### 🧠 B. Pembuatan dan Fine-Tuning RAG Model

5. **Pembuatan Chroma Database**
   - Menggunakan **embedding model** `LazarusNLP/all-indo-e5-small-v4`  
   - Chunk size: **500**  
   - Hasil database:
     ```
     chroma_uu_db_indo_v2
     ```

6. **Generate Dataset untuk Fine-Tuning Embedding Model**
   - Menggunakan skrip `generate_dataset_rag.py` dengan model **Llama 3.1:8B**.  
   - Format output dataset:
     ```json
     {
       "global_chunk_id": "",
       "text": "",
       "questions": ""
     }
     ```
   - Dari total **4.033 chunk**, setiap chunk dengan panjang lebih dari **50 token** digenerate sebanyak **3 pertanyaan**.  
   - Menghasilkan total **11.370 baris dataset** yang disimpan pada:
     ```
     training_model/dataset/generated_qa_dataset_v2.jsonl
     ```

7. **Fine-Tuning Embedding Model**
   - Menggunakan notebook `FT_Embedding_Models_on_Domain_Specific_Data.ipynb`.  
   - Base model: `LazarusNLP/all-indo-e5-small-v4`  
   - Hasil model fine-tune:
     ```
     rag_model/model/all-indo-e5-small-v4-matryoshka-v2
     ```

8. **Integrasi ke Pipeline RAG**
   - Semua komponen (base model, fine-tuned chat model, dan embedding model) digabungkan dalam satu pipeline:
     ```
     rag_model/main.py
     ```

---

## 📊 Hasil Akhir

- Chat model: `taxbot_v8.gguf`
- Embedding model: `all-indo-e5-small-v4-matryoshka-v2`
- Database: `chroma_uu_db_indo_v2`
- Final pipeline: `rag_model/main.py`

---