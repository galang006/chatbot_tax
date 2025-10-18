from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

model = AutoModelForCausalLM.from_pretrained("merged-taxbot-SeaLLMs-v3-1.5B-Chat-v6", device_map=device)
tokenizer = AutoTokenizer.from_pretrained("merged-taxbot-SeaLLMs-v3-1.5B-Chat-v6")

prompt = ""
messages = [
    {"role": "system", "content": "If the information is not relate to tax, please say 'Maaf, saya tidak memiliki pemahaman tentang hal itu' instead of making up facts!. Kamu adalah asisten ahli pajak Indonesia. Jawaban harus faktual dan menyebutkan sumber hukum (pasal/UU) di akhir. kamu hanya boleh menjawab pertanyaan seputar pajak. Jika pertanyaannya tidak terkait pajak dan informasi tidak ditemukan, jawab dengan: 'Maaf, saya tidak memiliki pemahaman tentang hal itu."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

print(f"Formatted text:\n {text}")
print(f"Model input:\n {model_inputs}")

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(f"Response:\n {response[0]}")
