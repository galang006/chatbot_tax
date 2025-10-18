import json
import subprocess
import re
from tqdm import tqdm
from pathlib import Path
import time
import ollama
import glob
from config import *

client = ollama.Client()

def run_ollama_client(prompt, temperature=0.7):
    try:
        response = client.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": temperature})
        return response.response.strip()
    except Exception as e:
        print(f"❌ Error Ollama client: {e}")
        return ""

def save_checkpoint(out_data):
    """Simpan checkpoint sementara"""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        for item in out_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def extract_questions(raw_text):
    questions = []
    pattern = r'^\s*(\d+)[\.\)]\s*(.+\?)\s*$'
    lines = raw_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            questions.append(match.group(2).strip())
            continue
        if line.endswith('?'):
            question = re.sub(r'^\s*[\d\-\*\•]+[\.\)]\s*', '', line)
            if question and len(question) > 10:
                questions.append(question)
    return questions[:5]

def generate_questions(isi, prompt_type="specific", max_retries=2):
    prompt = PROMPT_QUESTION_SPECIFIC.format(isi=isi) if prompt_type=="specific" else PROMPT_QUESTION_COMPLEX.format(isi=isi)
    for attempt in range(max_retries):
        raw = run_ollama_client(prompt, temperature=TEMP_QUESTION)
        if not raw:
            time.sleep(1)
            continue
        questions = extract_questions(raw)
        if questions:
            return questions
        time.sleep(1)
    return []

def generate_answer(isi, question, sumber, answer_type="specific", max_retries=2):
    prompt = PROMPT_ANSWER_SPECIFIC.format(isi=isi, question=question, sumber=sumber) if answer_type=="specific" else PROMPT_ANSWER_COMPLEX.format(isi=isi, question=question, sumber=sumber)
    for attempt in range(max_retries):
        answer = run_ollama_client(prompt, temperature=TEMP_ANSWER)
        if answer and len(answer) > 10:
            return answer
        time.sleep(1)
    return "Tidak dapat menghasilkan jawaban."

def save_jsonl(out_data, output_file=OUTPUT_FILE):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in out_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except:
        print("❌ Ollama tidak ditemukan, pastikan sudah serve dan model ter-pull.")
        return

    folders = [
        # (DATA_FOLDER_SPECIFIC, "specific"),
        (DATA_FOLDER_COMPLEX, "complex")
    ]

    out_data = []

    for folder_path, prompt_type in folders:
        json_files = glob.glob(f"{folder_path}/*.json")
        print(f"📂 Processing {prompt_type} questions from {len(json_files)} files in {folder_path}")
        
        for file_path in tqdm(json_files, desc=f"Processing {prompt_type} files"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for entry in data:
                isi = entry.get("Isi", "")
                if not isi:
                    continue
                
                sumber = ""
                if prompt_type == "specific":
                    sumber = f"{entry.get('UU','')} {entry.get('Pasal','')} {entry.get('Ayat','')} {entry.get('Sumber','')}".strip()
                else:
                    sumber = f"{entry.get('UU','')} {entry.get('Sumber','')}".strip()

                questions = generate_questions(isi, prompt_type=prompt_type)
                for q in questions:
                    a = generate_answer(isi, q, sumber, answer_type=prompt_type)
                    out_data.append({
                        "instruction": q,
                        "output": a,
                        "source": sumber,
                        "type": prompt_type
                    })

                    # Simpan checkpoint setiap 10 Q&A
                    if len(out_data) % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(out_data)
    
    # Simpan final output
    save_jsonl(out_data)
    print(f"\n✅ Generated {len(out_data)} Q&A pairs saved to {OUTPUT_FILE}")
    # Hapus checkpoint kalau sudah selesai
    Path(CHECKPOINT_FILE).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
