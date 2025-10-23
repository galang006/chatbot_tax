import json
import random

input_file = "dataset/generated_question_dataset.jsonl" 
output_file = "selected_questions_v3.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

spesifik_questions = [d["question_spesifik"] for d in data if d.get("question_spesifik")]
studi_kasus_questions = [d["question_studi_kasus"] for d in data if d.get("question_studi_kasus")]

print(f"📚 Total pertanyaan spesifik: {len(spesifik_questions)}")
print(f"📚 Total pertanyaan studi kasus: {len(studi_kasus_questions)}")

selected_spesifik = random.sample(spesifik_questions, min(50, len(spesifik_questions)))
selected_studi = random.sample(studi_kasus_questions, min(50, len(studi_kasus_questions)))

selected_data = (
    [{"type": "spesifik", "prompt": q} for q in selected_spesifik] +
    [{"type": "studi_kasus", "prompt": q} for q in selected_studi]
)

with open(output_file, "w", encoding="utf-8") as f:
    for item in selected_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Berhasil menyimpan {len(selected_data)} pertanyaan ke '{output_file}'")
print(f"- {len(selected_spesifik)} pertanyaan spesifik")
print(f"- {len(selected_studi)} pertanyaan studi kasus")
