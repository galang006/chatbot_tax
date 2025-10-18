from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
import os

CHROMA_PATH = "database/chroma_uu_db_indo"

embedding_function = HuggingFaceEmbeddings(
    model_name="LazarusNLP/all-indo-e5-small-v4",
    model_kwargs={"device": "cpu"},
)

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function
)

def preview_chunks(limit=5, save_to_json=False):
    """
    Menampilkan beberapa chunk dari database Chroma dan opsional menyimpannya ke JSON.

    Args:
        limit (int): Jumlah chunk yang ingin ditampilkan.
        save_to_json (bool): Jika True, akan menyimpan semua chunk ke file JSON.
    """
    data = db.get(limit=limit if not save_to_json else None)

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    print(f"\n📚 Jumlah chunk yang diambil: {len(documents)}\n")

    for i, (content, meta) in enumerate(zip(documents[:limit], metadatas[:limit])):
        print(f"--- Chunk {i+1} ---")
        print(f"UU: {meta.get('uu', '-')}")
        print(f"BAB: {meta.get('bab', '-')}")
        print(f"Pasal: {meta.get('pasal', '-')}")
        print(f"Ayat: {meta.get('ayat', '-')}")
        print(f"Sumber: {meta.get('sumber', '-')}")
        print(f"\nIsi Chunk (awal):\n{content[:500]}...")
        print("-" * 80)

    if save_to_json:
        chunks_data = [
            {"content": c, "metadata": m} for c, m in zip(documents, metadatas)
        ]
        output_path = os.path.join(CHROMA_PATH, "exported_chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Semua chunk berhasil disimpan ke: {output_path}")


if __name__ == "__main__":
    preview_chunks(limit=5, save_to_json=False)
