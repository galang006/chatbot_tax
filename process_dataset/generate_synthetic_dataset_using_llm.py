import json
import subprocess
import re
from tqdm import tqdm
from pathlib import Path
import time
import ollama

filename = "80_BAB_VIII_KETENTUAN_PERALIHAN"
input_file = f"dataset/output_extract_pdf/{filename}.json"
output_file = f"dataset/qa_dataset/{filename}.jsonl"
model_name = "gemma3:4b"

client = ollama.Client()

PROMPT_QUESTION = """Buat 5 pertanyaan berbeda dalam bahasa Indonesia berdasarkan isi pasal berikut.
Pertanyaan harus terdengar natural, seperti orang awam bertanya.
⚠️ Jangan ulang kata yang persis ada di teks. 
Tuliskan pertanyaan dalam format:
1. ...
2. ...
3. ...
4. ...
5. ...

Isi Pasal:
{isi}

Pertanyaan:"""

PROMPT_ANSWER = """Jawablah pertanyaan berikut berdasarkan isi pasal yang diberikan.

ATURAN:
- Jawaban harus faktual dan to the point
- Sertakan sumber pasal di akhir kalimat dengan cara yang natural, misal: "sesuai dengan Pasal 38 UU Nomor 6 Tahun 2023".
- Gunakan bahasa formal

Isi Pasal:
{isi}

Sumber Pasal:
{sumber_uu}

Pertanyaan: {question}

Jawaban:"""

def run_ollama_client(prompt, temperature=0.7):
    """Generate text menggunakan Ollama client"""
    try:
        response = client.generate(model=model_name, prompt=prompt, options={"temperature": temperature} )
        return response.response.strip()
    except Exception as e:
        print(f"❌ Error Ollama client: {e}")
        return ""

def extract_questions(raw_text):
    """
    Extract pertanyaan dari output model
    Mencari pattern: 1. ... ? atau angka. text?
    """
    questions = []
    
    # Pattern 1: Numbered list dengan tanda tanya
    # Contoh: 1. Apa itu pajak?
    pattern = r'^\s*(\d+)[\.\)]\s*(.+\?)\s*$'
    
    lines = raw_text.split('\n')
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Try numbered pattern
        match = re.match(pattern, line)
        if match:
            question = match.group(2).strip()
            questions.append(question)
            continue
        
        # Fallback: Any line ending with ?
        if line.endswith('?'):
            # Remove leading numbers/bullets
            question = re.sub(r'^\s*[\d\-\*\•]+[\.\)]\s*', '', line)
            if question and len(question) > 10:
                questions.append(question)
    
    return questions[:5]

def generate_questions(isi, max_retries=2):
    """
    Generate 5 pertanyaan dari isi pasal
    
    Args:
        isi: Isi pasal/ayat
        max_retries: Jumlah retry jika gagal
    
    Returns:
        List of questions (max 5)
    """
    prompt = PROMPT_QUESTION.format(isi=isi)
    
    for attempt in range(max_retries):
        raw = run_ollama_client(prompt, temperature=0.9)
        
        if not raw:
            print(f"  ⚠️ Attempt {attempt + 1}: No output from model")
            time.sleep(1)
            continue
        
        questions = extract_questions(raw)
        
        if questions:
            print(f"  ✅ Generated {len(questions)} questions")
            return questions
        else:
            print(f"  ⚠️ Attempt {attempt + 1}: No questions extracted")
            print(f"  Raw output: {raw[:100]}...")
            time.sleep(1)
    
    print("  ❌ Failed to generate questions after retries")
    return []

def generate_answer(isi, question, sumber_uu, max_retries=2):
    """
    Generate jawaban untuk pertanyaan berdasarkan isi pasal
    
    Args:
        isi: Isi pasal/ayat
        question: Pertanyaan yang akan dijawab
        max_retries: Jumlah retry jika gagal
    
    Returns:
        String jawaban
    """
    prompt = PROMPT_ANSWER.format(isi=isi, question=question, sumber_uu=sumber_uu)
    
    for attempt in range(max_retries):
        answer = run_ollama_client(prompt, temperature=0.2)
        
        if answer and len(answer) > 10:
            return answer
        
        print(f"    ⚠️ Attempt {attempt + 1}: Answer too short or empty")
        time.sleep(1)
    
    return "Tidak dapat menghasilkan jawaban."

def save_checkpoint(out_data, checkpoint_file="checkpoint.jsonl"):
    """Save progress checkpoint"""
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        for item in out_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ File tidak ditemukan: {input_file}")
        return
    
    # Load data
    print(f"📂 Loading data from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"📊 Total entries: {len(data)}")
    
    # Check if Ollama is available
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama tidak ditemukan.")
        return
    
    # Process data
    out_data = []
    failed_count = 0
    
    for idx, row in enumerate(tqdm(data, desc="Generating Q&A"), 1):
        isi = row.get("isi", "")
        sumber = row.get("sumber", "")
        
        if not isi:
            print(f"  ⚠️ Entry {idx}: Empty isi, skipping")
            continue
        
        print(f"\n📝 Processing entry {idx}/{len(data)}")
        print(f"   Source: {sumber}")
        
        # Generate questions
        questions = generate_questions(isi)
        
        if not questions:
            failed_count += 1
            continue
        
        # Generate answers for each question
        for q_idx, question in enumerate(questions, 1):
            print(f"   Q{q_idx}: {question[:60]}...")
            
            answer = generate_answer(isi, question, sumber)
            print(f"   A{q_idx}: {answer[:60]}...")
            
            item = {
                "instruction": question,
                "output": answer,
                "source": sumber
            }
            out_data.append(item)
        
        # Save checkpoint every 10 entries
        if idx % 5 == 0:
            save_checkpoint(out_data, "checkpoint_temp.jsonl")
            print(f"💾 Checkpoint saved ({len(out_data)} items)")
    
    # Save final output
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in out_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Print summary
    print("\n" + "="*60)
    print("✅ SELESAI!")
    print("="*60)
    print(f"📄 Input entries: {len(data)}")
    print(f"❌ Failed entries: {failed_count}")
    print(f"✅ Total Q&A pairs: {len(out_data)}")
    print(f"📊 Average Q&A per entry: {len(out_data) / (len(data) - failed_count):.1f}")
    print(f"💾 Output file: {output_file}")
    
    # Sample output
    if out_data:
        print("\n📋 Sample output:")
        sample = out_data[0]
        print(f"   Instruction: {sample['instruction']}")
        print(f"   Output: {sample['output'][:100]}...")
        print(f"   Source: {sample['source']}")

def test_ollama_connection():
    """Test if Ollama is working"""
    print("🧪 Testing Ollama connection...")
    
    test_prompt = "Halo, apa kabar?"
    result = run_ollama_client(test_prompt, temperature=0.5)
    
    if result:
        print(f"✅ Ollama working! Response: {result[:100]}")
        return True
    else:
        print("❌ Ollama not responding")
        return False

if __name__ == "__main__":
    # Test connection first
    if not test_ollama_connection():
        print("\n💡 Tips:")
        print("1. Pastikan Ollama sudah diinstall")
        print("2. Jalankan: ollama serve")
        print("3. Pull model: ollama pull gemma3:4b")
        exit(1)
    
    print("\n" + "="*60)
    main()