import os
import re
import json
from pdfminer.high_level import extract_text
from pathlib import Path

PDF_PATH = r"dataset\SDSN_2023.pdf"
OUTPUT_DIR = "output"
SKIP_PAGES = 6  

UU_MAPPING = {
    "*": "UU Nomor 9 Tahun 1994",
    "**": "UU Nomor 16 Tahun 2000",
    "***": "UU Nomor 28 Tahun 2007",
    "****": "UU Nomor 16 Tahun 2009",
    "*****": "UU Nomor 11 Tahun 2020",
    "******": "UU Nomor 7 Tahun 2021",
    "*******": "UU Nomor 6 Tahun 2023"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_uu_from_stars(star_text):
    """Konversi bintang ke nama UU"""
    match = re.search(r'(\*+)\)', star_text)
    if match:
        stars = match.group(1)
        return UU_MAPPING.get(stars, "")
    return ""

def clean_text(text):
    """Bersihkan whitespace berlebih"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_bab_info(text, position):
    """Extract informasi BAB di posisi tertentu"""
    pattern = r'BAB\s+([IVXLCDM]+[A-Z]*)\s+([^\n]+)'
    
    match = re.search(pattern, text[position:position+500])
    if match:
        bab_roman = match.group(1).strip()
        bab_title = match.group(2).strip()
        return bab_roman, bab_title
    return None, None

def split_into_babs(text):
    """
    Split text menjadi BAB-BAB dengan tracking nomor urut global
    """
    bab_pattern = r'BAB\s+([IVXLCDM]+[A-Z]*)\s+'
    
    bab_matches = list(re.finditer(bab_pattern, text))
    
    if not bab_matches:
        return []
    
    babs = []
    global_counter = 1
    
    for i, match in enumerate(bab_matches):
        start = match.start()
        end = bab_matches[i + 1].start() if i + 1 < len(bab_matches) else len(text)
        
        bab_roman = match.group(1).strip()
        
        title_start = match.end()
        title_end = text.find('\n', title_start)
        bab_title = text[title_start:title_end].strip() if title_end > title_start else ""
        
        bab_content = text[start:end]
        
        babs.append({
            'counter': global_counter,
            'roman': bab_roman,
            'title': bab_title,
            'content': bab_content
        })
        
        global_counter += 1
    
    return babs

def extract_pasal_ayat(bab_content):
    """
    Extract PASAL dan AYAT dari content BAB
    Menangani:
    - Numbering: 1., 2. atau (1), (2)
    - Sub-items: a., b., c. setelah ***)
    - Penjelasan Pasal
    - PASAL dengan bintang
    """
    entries = []
    
    pasal_pattern = r'(?=PASAL\s+\d+[A-Z]*)'
    pasal_blocks = re.split(pasal_pattern, bab_content, flags=re.IGNORECASE)
    
    for block in pasal_blocks:
        if not block.strip():
            continue
        
        pasal_match = re.search(r'PASAL\s+(\d+[A-Z]*)', block, re.IGNORECASE)
        if not pasal_match:
            continue
        
        pasal_num = pasal_match.group(1)
        content_start = pasal_match.end()
        
        pasal_line_end = block.find('\n', content_start)
        pasal_line = block[content_start:pasal_line_end] if pasal_line_end > 0 else ""
        pasal_has_stars = '***' in pasal_line or '**' in pasal_line or '*' in pasal_line
        
        pasal_uu = get_uu_from_stars(pasal_line) if pasal_has_stars else ""
        
        pasal_content = block[content_start:].strip()
        
        if pasal_content.startswith("Penjelasan Pasal") or "Penjelasan Pasal" in pasal_content[:50]:
            penjelasan_end = re.search(r'PASAL\s+\d+', pasal_content[20:], re.IGNORECASE)
            if penjelasan_end:
                penjelasan_text = pasal_content[:20 + penjelasan_end.start()].strip()
            else:
                penjelasan_text = pasal_content.strip()
            
            penjelasan_clean = re.sub(r'\*+\)', '', penjelasan_text)
            penjelasan_clean = clean_text(penjelasan_clean)
            
            entries.append({
                "isi": penjelasan_clean,
                "sumber": "" 
            })
            continue
        
        lines = pasal_content.split('\n')
        current_ayat = None
        current_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            ayat_paren_match = re.match(r'^\((\d+[a-z]*)\)', line)
            ayat_number_match = re.match(r'^(\d+)\.\s+', line)
            
            is_new_ayat = ayat_paren_match or ayat_number_match
            
            if is_new_ayat:
                if current_ayat and current_content:
                    save_ayat_entry(entries, pasal_num, current_ayat, current_content, pasal_uu)
                
                current_ayat = ayat_paren_match.group(1) if ayat_paren_match else ayat_number_match.group(1)
                
                line_content = re.sub(r'^\(?\d+[a-z]*\)?\.?\s*', '', line)
                current_content = [line_content]
            else:
                current_content.append(line)
            
            i += 1
        
        if current_ayat and current_content:
            save_ayat_entry(entries, pasal_num, current_ayat, current_content, pasal_uu)
        
        if not current_ayat and current_content:
            full_text = '\n'.join(current_content)
            
            uu_source = get_uu_from_stars(full_text)
            if not uu_source and pasal_uu:
                uu_source = pasal_uu
            
            clean_content = re.sub(r'\*+\)', '', full_text)
            clean_content = clean_text(clean_content)
            
            if clean_content and len(clean_content) > 5:
                sumber = f"Pasal {pasal_num} {uu_source}" if uu_source else ""
                entries.append({
                    "isi": clean_content,
                    "sumber": sumber
                })
    
    return entries

def save_ayat_entry(entries, pasal_num, ayat_num, content_lines, pasal_uu=""):
    """
    Save ayat entry dengan handling:
    - Stars untuk detect UU
    - Sub-items (a., b., c.) tetap dalam satu entry
    """
    full_text = '\n'.join(content_lines)
    
    has_stars = '***' in full_text or '**' in full_text or '*' in full_text
    
    uu_source = get_uu_from_stars(full_text)
    
    if not uu_source and pasal_uu:
        uu_source = pasal_uu
    
    star_pos = full_text.rfind('***)')
    if star_pos == -1:
        star_pos = full_text.rfind('**)')
    if star_pos == -1:
        star_pos = full_text.rfind('*)')
    
    if star_pos > 0:
        after_stars = full_text[star_pos+4:].strip()
        
        if re.match(r'^[a-z]\.\s', after_stars):
            clean_content = re.sub(r'\*+\)', '', full_text)
        else:
            clean_content = full_text[:star_pos]
            clean_content = re.sub(r'\*+\)', '', clean_content)
    else:
        clean_content = full_text
    
    clean_content = clean_text(clean_content)
    
    sumber = ""
    if has_stars and uu_source:
        sumber = f"Pasal {pasal_num} Ayat {ayat_num} {uu_source}"
    
    if clean_content and len(clean_content) > 5:
        entries.append({
            "isi": clean_content,
            "sumber": sumber
        })

def sanitize_filename(name):
    """Clean filename"""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:100]

def process_pdf(pdf_path):
    """Main processing function"""
    
    print(f"📄 Reading PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    try:
        full_text = extract_text(pdf_path)
        
        print(f"✅ Text extracted: {len(full_text)} characters")
        
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return
    
    start_pattern = r'BAB\s+I\s+KETENTUAN\s+UMUM'
    start_match = re.search(start_pattern, full_text, re.IGNORECASE)
    
    if start_match:
        content_text = full_text[start_match.start():]
        print(f"✅ Found content start at position {start_match.start()}")
    else:
        print("⚠️  Could not find BAB I, processing entire document")
        content_text = full_text
    
    babs = split_into_babs(content_text)
    print(f"📚 Found {len(babs)} BABs")
    
    for bab in babs:
        counter = bab['counter']
        roman = bab['roman']
        title = bab['title']
        content = bab['content']
        
        print(f"\n📖 Processing BAB {counter}: {roman} - {title}")
        
        entries = extract_pasal_ayat(content)
        
        if not entries:
            print(f"   ⚠️  No entries extracted")
            continue
        
        filename = f"{counter:02d}_BAB_{roman}_{sanitize_filename(title)}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ Saved: {filename} ({len(entries)} entries)")
    
    print(f"\n✅ Processing complete! Files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_pdf(PDF_PATH)