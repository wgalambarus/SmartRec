"""
smart_recruiter.py
Single-file pipeline:
- ekstrak teks dari PDF / DOCX (opsional OCR)
- ekstrak fitur dasar (email, phone, education, years)
- ambil pengalaman sections (heuristic)
- compute semantic similarity antara pengalaman dan job description (sentence-transformers)
- scoring rule-based + semantic relevance
- ranking dan simpan hasil ke CSV/JSON

Usage:
python smart_recruiter.py --cv_folder ./cvs --job_desc "Data Scientist with ML & Python experience" --out results.csv
"""

import os
import re
import json
import argparse
from datetime import datetime
from glob import glob
from collections import defaultdict

# file reading
import docx2txt
import fitz  # PyMuPDF

# optional OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# embeddings
from sentence_transformers import SentenceTransformer, util

# Data output
import pandas as pd

# -----------------------
# Utilities: Text Extraction
# -----------------------
def extract_text_from_pdf(path, try_ocr_if_empty=False):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    text = text.strip()
    if not text and try_ocr_if_empty and OCR_AVAILABLE:
        # fallback to OCR each page as image
        imgs = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                imgs.append(Image.open(io.BytesIO(img_bytes)))
        for im in imgs:
            text += pytesseract.image_to_string(im) + "\n"
    return text

def extract_text(path, try_ocr_if_empty=False):
    path = str(path)
    lower = path.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(path, try_ocr_if_empty=try_ocr_if_empty)
    elif lower.endswith(".docx"):
        return docx2txt.process(path)
    elif lower.endswith((".png", ".jpg", ".jpeg", ".tiff")):
        if not OCR_AVAILABLE:
            raise RuntimeError("pytesseract not available for OCR. Install tesseract and pytesseract.")
        return pytesseract.image_to_string(Image.open(path))
    else:
        raise ValueError("Unsupported file format: " + path)

# -----------------------
# Basic regex extractors
# -----------------------
def extract_email(text):
    m = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return m[0] if m else None

def extract_phone(text):
    # common phone patterns, rough
    m = re.findall(r'(\+?\d[\d\s\-\(\)]{6,}\d)', text)
    if m:
        # return most likely (longest)
        return max(m, key=len)
    return None

def extract_dob_and_age(text):
    patterns = [
        r'(\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4})',  # dd/mm/yyyy or dd-mm-yyyy
        r'(\d{4}[\/\-\s]\d{1,2}[\/\-\s]\d{1,2})',    # yyyy-mm-dd
        r'Date of Birth[:\s\-]*([A-Za-z0-9 ,\-\/]+)'
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            s = m.group(1)
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y"):
                try:
                    dob = datetime.strptime(s.strip(), fmt)
                    age = datetime.now().year - dob.year
                    return dob.date().isoformat(), age
                except Exception:
                    continue
    return None, None

def extract_education_level(text):
    # priority: S3 > S2 > S1 > D3 > SMA
    mapping = [
        (r'\b(PhD|S3|Doctor)\b', 'S3'),
        (r'\b(Master|S2|Magister|M.Sc|M\.S|MEng)\b', 'S2'),
        (r'\b(Bachelor|S1|B\.Sc|BSc|BA|Sarjana)\b', 'S1'),
        (r'\b(Diploma|D3)\b', 'D3'),
        (r'\b(High School|SMA|SMK)\b', 'SMA'),
    ]
    for pat, lvl in mapping:
        if re.search(pat, text, re.IGNORECASE):
            return lvl
    return "Unknown"

def extract_years_experience(text):
    # heuristic: find years like 2016, 2020 etc and compute span
    years = re.findall(r'(20[0-3]\d|19[7-9]\d)', text)
    years = sorted({int(y) for y in years})
    if len(years) >= 2:
        span = years[-1] - years[0]
        if 0 <= span <= 60:
            return span
    # fallback: find durations like "X years"
    m = re.search(r'(\d{1,2})\s+years?', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

# -----------------------
# Experience and skills extraction (heuristic)
# -----------------------
def extract_experience_sections(text):
    # naive: split by common headings
    headings = ['experience', 'work experience', 'employment history', 'professional experience', 'experience:', 'work history']
    # split lines and find lines with headings
    lines = text.splitlines()
    text_lower = text.lower()
    start_idx = None
    for i, l in enumerate(lines):
        if any(h in l.lower() for h in headings):
            start_idx = i
            break
    if start_idx is None:
        # fallback: try to capture paragraphs that contain years or job-like words
        paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        candidates = [p for p in paras if re.search(r'(engineer|developer|analyst|manager|worked|experience|intern)', p, re.IGNORECASE)]
        return candidates[:5]  # limited number
    # capture next few lines as experience block
    block = "\n".join(lines[start_idx:start_idx+40])
    # split by bullet or linebreaks into entries
    entries = [e.strip() for e in re.split(r'[\n•\-•\u2022]+', block) if e.strip()]
    return entries[:20]

def extract_skills_keyword(text, skill_list=None):
    # default skill list (extendable)
    if skill_list is None:
        skill_list = [
            'python','machine learning','deep learning','nlp','sql','excel','power bi','tableau',
            'pandas','numpy','scikit-learn','tensorflow','pytorch','java','javascript','react','laravel',
            'docker','kubernetes','aws','azure','gcp','communication','leadership'
        ]
    found = set()
    for s in skill_list:
        if re.search(r'\b' + re.escape(s) + r'\b', text, re.IGNORECASE):
            found.add(s)
    return sorted(found)

# -----------------------
# Semantic model (sentence-transformers)
# -----------------------
EMBED_MODEL = None
def load_embedding_model(name="all-MiniLM-L6-v2"):
    global EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = SentenceTransformer(name)
    return EMBED_MODEL

def semantic_similarity_score(text_chunks, job_desc, model):
    # text_chunks: list of strings (candidate experience entries)
    if not text_chunks:
        return 0.0, 0.0  # (max_sim, avg_sim)
    # encode all
    emb_job = model.encode(job_desc, convert_to_tensor=True)
    emb_exp = model.encode(text_chunks, convert_to_tensor=True)
    sims = util.cos_sim(emb_exp, emb_job)  # shape (n,1)
    sims = sims.cpu().numpy().reshape(-1)
    max_sim = float(sims.max())
    avg_sim = float(sims.mean())
    return max_sim, avg_sim

# -----------------------
# Scoring logic (rule-based with semantic)
# -----------------------
def score_candidate(parsed, job_desc, model, weights=None):
    """
    parsed: dict with keys: 'text', 'email', 'phone', 'dob','age','education','years_exp','skills','experience_entries'
    job_desc: string
    model: embedding model
    weights: dict overrides
    returns: (score, breakdown dict)
    """
    if weights is None:
        weights = {
            "education": 15,
            "years": 10,
            "skill_each": 6,
            "semantic_max": 40,
            "semantic_avg": 10,
            "age_bonus": 4
        }
    score = 0.0
    breakdown = {}
    # education
    edu = parsed.get("education", "Unknown")
    edu_score_map = {"S3": 15, "S2": 12, "S1": 8, "D3": 4, "SMA": 1, "Unknown": 0}
    edu_points = edu_score_map.get(edu, 0)
    score += edu_points
    breakdown['education'] = edu_points

    # years of experience
    years = parsed.get("years_exp") or 0
    years_points = min(years * 1.5, weights["years"])  # cap
    score += years_points
    breakdown['years'] = years_points

    # skills: overlap with job description keywords (also treat as positive)
    skills = parsed.get("skills", [])
    jd_lower = job_desc.lower()
    matched = 0
    for s in skills:
        # if skill appears in job desc OR generic
        if s.lower() in jd_lower:
            matched += 1
    skill_points = matched * weights["skill_each"]
    score += skill_points
    breakdown['skill_points'] = skill_points
    breakdown['skill_matches'] = matched

    # semantic similarity on experience entries
    entries = parsed.get("experience_entries", [])
    max_sim, avg_sim = semantic_similarity_score(entries, job_desc, model)
    # scale sims (they are 0..1). Multiply to weights
    sim_points = max_sim * weights["semantic_max"] + avg_sim * weights["semantic_avg"]
    score += sim_points
    breakdown['semantic_max'] = round(max_sim,3)
    breakdown['semantic_avg'] = round(avg_sim,3)
    breakdown['semantic_points'] = round(sim_points,3)

    # age bonus (optional): prefer mid-career 22..40
    age = parsed.get("age")
    age_points = 0
    if isinstance(age, int) and 22 <= age <= 40:
        age_points = weights["age_bonus"]
        score += age_points
    breakdown['age_points'] = age_points

    # normalize / final
    breakdown['raw_score'] = round(score,3)
    return score, breakdown

# -----------------------
# High level per-CV parse function
# -----------------------
def parse_cv_file(path, try_ocr=False, embed_model=None):
    txt = extract_text(path, try_ocr_if_empty=try_ocr)
    txt = txt.replace('\r','\n')
    parsed = {}
    parsed['file'] = os.path.basename(path)
    parsed['text'] = txt
    parsed['email'] = extract_email(txt)
    parsed['phone'] = extract_phone(txt)
    parsed['dob'], parsed['age'] = extract_dob_and_age(txt)
    parsed['education'] = extract_education_level(txt)
    parsed['years_exp'] = extract_years_experience(txt)
    parsed['skills'] = extract_skills_keyword(txt)
    parsed['experience_entries'] = extract_experience_sections(txt)
    # name heuristic: first non-empty line that doesn't contain CV word, LENGTH limit
    first_lines = [l.strip() for l in txt.splitlines() if l.strip()]
    name = None
    for l in first_lines[:6]:
        if not re.search(r'curriculum vitae|cv|resume|profile|contact', l, re.IGNORECASE) and len(l.split()) <= 5:
            name = l
            break
    parsed['name'] = name or "Unknown"
    return parsed

# -----------------------
# Process folder and rank
# -----------------------
def process_and_rank(cv_folder, job_desc, try_ocr=False, out_csv=None, out_json=None):
    paths = []
    for ext in ("pdf","docx","DOCX","PDF"):
        paths.extend(glob(os.path.join(cv_folder, f'**/*.{ext}'), recursive=True))
    if not paths:
        raise FileNotFoundError("No CV files found in folder: " + cv_folder)

    model = load_embedding_model()
    results = []
    for p in paths:
        parsed = parse_cv_file(p, try_ocr=try_ocr, embed_model=model)
        score, breakdown = score_candidate(parsed, job_desc, model)
        parsed_summary = {
            "file": parsed['file'],
            "name": parsed['name'],
            "email": parsed['email'],
            "phone": parsed['phone'],
            "age": parsed['age'],
            "education": parsed['education'],
            "years_exp": parsed['years_exp'],
            "skills": ";".join(parsed['skills']),
            "score": round(score,3),
            "breakdown": breakdown
        }
        results.append(parsed_summary)

    # sort desc
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

    if out_csv:
        df = pd.DataFrame([
            {**{k:v for k,v in r.items() if k!='breakdown'}, **{"breakdown": json.dumps(r['breakdown'])}} for r in results_sorted
        ])
        df.to_csv(out_csv, index=False)
    if out_json:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    return results_sorted

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Smart Recruitment: CV ranking by semantic relevance")
    parser.add_argument("--cv_folder", required=True, help="Folder containing CVs (.pdf, .docx)")
    parser.add_argument("--job_desc", required=True, help="Job description text to match candidates against")
    parser.add_argument("--out", default="ranking.csv", help="Output CSV path")
    parser.add_argument("--json", default=None, help="Output JSON path (optional)")
    parser.add_argument("--ocr", action="store_true", help="Use OCR fallback for empty PDFs / images (requires pytesseract)")
    args = parser.parse_args()

    print("Loading embedding model (this may take a few seconds)...")
    load_embedding_model()
    print("Processing CVs in:", args.cv_folder)
    results = process_and_rank(args.cv_folder, args.job_desc, try_ocr=args.ocr, out_csv=args.out, out_json=args.json)
    print("Done. Top candidates:")
    for i, r in enumerate(results[:10], 1):
        print(f"{i}. {r['name']}  ({r['file']})  score={r['score']}  skills={r['skills']}")

if __name__ == "__main__":
    main()
