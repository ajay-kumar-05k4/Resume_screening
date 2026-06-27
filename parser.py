"""
Resume Parser Module
Extracts text from PDF resumes and preprocesses the content.
"""

import io
import re
import os
import nltk
import pdfplumber
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── NLTK data download (Streamlit Cloud safe) ──────────────────────────────────
_NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(_NLTK_DIR, exist_ok=True)          # never raises FileExistsError

for _resource, _path in [
    ("punkt",      "tokenizers/punkt"),
    ("punkt_tab",  "tokenizers/punkt_tab"),
    ("stopwords",  "corpora/stopwords"),
]:
    try:
        nltk.data.find(_path)
    except (LookupError, OSError):
        try:
            nltk.download(_resource, download_dir=_NLTK_DIR, quiet=True)
        except Exception:
            pass   # best-effort; missing data will surface later with a clear error


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file path."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


def extract_text_from_file(file_object) -> str:
    """Extract text from a Streamlit-uploaded file object."""
    try:
        raw = file_object.read()
        if not raw:
            print("Error: uploaded file is empty.")
            return ""
        pdf_file = io.BytesIO(raw)
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading uploaded file: {e}")
        return ""


def preprocess_text(text: str) -> str:
    """Clean and normalise text."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z0-9\s+\-#.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(resume_text: str, skills_list: list) -> list:
    """Extract skills from resume text by matching against a predefined list."""
    if not resume_text or not skills_list:
        return []
    resume_lower = resume_text.lower()
    found = set()
    for skill in skills_list:
        skill_lower = skill.lower()
        pattern = r"\b" + re.escape(skill_lower) + r"\b"
        if re.search(pattern, resume_lower):
            found.add(skill_lower)
    return list(found)


def extract_contact_info(resume_text: str) -> dict:
    """Extract contact information from resume text."""
    contact_info: dict = {}
    if not resume_text:
        return contact_info

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    emails = re.findall(email_pattern, resume_text)
    if emails:
        contact_info["email"] = emails[0]

    phone_pattern = r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    phones = re.findall(phone_pattern, resume_text)
    if phones:
        contact_info["phone"] = phones[0]

    linkedin_pattern = r"linkedin\.com/in/[\w\-]+"
    linkedin = re.findall(linkedin_pattern, resume_text, re.IGNORECASE)
    if linkedin:
        contact_info["linkedin"] = linkedin[0]

    return contact_info


def tokenize_and_filter(text: str) -> list:
    """Tokenize text and remove English stopwords."""
    if not text:
        return []
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words("english"))
        return [t for t in tokens if t.isalnum() and t not in stop_words]
    except Exception:
        # Fallback: simple whitespace split if NLTK data unavailable
        return [w for w in text.lower().split() if w.isalnum()]
