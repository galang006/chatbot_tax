from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "database/chroma_uu_db_indo"  

embedding_function = HuggingFaceEmbeddings(
    model_name="LazarusNLP/all-indo-e5-small-v4",
    model_kwargs={"device": "cpu"},
)

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def rag_retrieve(query, k=3):
    print(f"\nQuery: {query}")
    results = db.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Hasil {i+1} (score={score}) ---")
        print(doc.metadata.get('sumber', '')) 
        print(doc.page_content[:500]) 
        print("-" * 50)

if __name__ == "__main__":
    while True:
        query = input("\nMasukkan query (atau 'exit' untuk keluar): ")
        if query.lower() == "exit":
            break
        rag_retrieve(query)
