from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface  import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import json
import os
import shutil

CHROMA_PATH = "database/chroma_uu_db_indo"
FOLDER_PATH = "/home/ubuntu/projek_chatbot_galang/process_dataset/dataset/uu_per_ayat"  

def load_documents():
    documents = []
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(FOLDER_PATH, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data_list = json.load(f) 
                for data in data_list:     
                    text = f"""
                    Isi: {data.get('Isi', '')}
                    Penjelasan: {data.get('Penjelasan', '')}
                    """
                    doc = Document(
                        page_content=text,
                        metadata={
                            "uu": data.get("UU", ""),
                            "bab": data.get("BAB", ""),
                            "pasal": data.get("Pasal", ""),
                            "ayat": data.get("Ayat", ""),
                            "sumber": data.get("Sumber", ""),
                        }
                    )
                    documents.append(doc)
    print(f"Loaded {len(documents)} documents from {FOLDER_PATH}")
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="LazarusNLP/all-indo-e5-small-v4"
    )

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to Chroma DB at '{CHROMA_PATH}'")

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
