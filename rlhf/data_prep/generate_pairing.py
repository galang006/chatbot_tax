import json
import itertools
import os
import time

# ===== Konfigurasi =====
INPUT_FILE = "dataset/generated_responses_v3.jsonl"
OUTPUT_FILE = "dataset/paired_responses_v3.jsonl"

# ===== Fungsi bantu =====
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def load_responses(filename):
    """Load semua jawaban dari file JSONL"""
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def group_by_question(data):
    """Kelompokkan semua jawaban berdasarkan pertanyaan"""
    grouped = {}
    for item in data:
        q = item["question"]
        grouped.setdefault(q, []).append(item)
    return grouped

def save_checkpoint(record, filename):
    """Simpan 1 record ke file checkpoint"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_completed_pairs(filename):
    """Ambil daftar pair (question + index) yang sudah selesai"""
    if not os.path.exists(filename):
        return set()
    completed = set()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                key = (rec["question"], rec["pair_index"])
                completed.add(key)
            except:
                continue
    return completed

# ===== Main Process =====
def main():
    print("🔍 Memuat hasil generate jawaban...")
    data = load_responses(INPUT_FILE)
    grouped = group_by_question(data)
    completed = load_completed_pairs(OUTPUT_FILE)

    total_pairs = 0
    for q, answers in grouped.items():
        if len(answers) < 2:
            continue
        total_pairs += len(list(itertools.combinations(answers, 2)))
    print(f"📚 Total pertanyaan: {len(grouped)} | Total pasangan (5C2 per pertanyaan): {total_pairs}")
    time.sleep(1)

    for q_idx, (question, answers) in enumerate(grouped.items(), start=1):
        if len(answers) < 2:
            continue

        pairs = list(itertools.combinations(answers, 2))
        for pair_idx, (a, b) in enumerate(pairs):
            key = (question, pair_idx)
            if key in completed:
                continue

            clear_screen()
            print(f"===== Pair {pair_idx + 1}/{len(pairs)} untuk pertanyaan {q_idx} =====\n")
            print(f"🟩 Context:\n{a['context']}\n")
            print(f"📜 Sumber: {a['source']}\n")
            print(f"❓ Question: {question}\n")
            print("──────────────────────────────────────────────")
            print("🔹 Answer A:")
            print(a["answer"].strip())
            print("──────────────────────────────────────────────")
            print("🔸 Answer B:")
            print(b["answer"].strip())
            print("──────────────────────────────────────────────\n")
            print("Pilih jawaban yang **lebih baik / lebih relevan** (0 = A lebih baik, 1 = B lebih baik)")
            
            while True:
                choice = input("Masukkan pilihan (0/1): ").strip()
                if choice in ["0", "1"]:
                    break
                print("❌ Input tidak valid! Harus 0 atau 1.")

            record = {
                "pair_index": pair_idx,
                "context": a["context"],
                "source": a["source"],
                "question": question,
                "answer_a": a["answer"],
                "answer_b": b["answer"],
                "answer_b_preferred": int(choice)
            }

            save_checkpoint(record, OUTPUT_FILE)
            print("✅ Pair disimpan. Melanjutkan ke pasangan berikutnya...")
            time.sleep(1)

    print("\n🎯 Semua pair selesai diproses dan disimpan ke", OUTPUT_FILE)

if __name__ == "__main__":
    main()
