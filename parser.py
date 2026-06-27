"""
Resume Parser Module
Extracts text from PDF resumes and preprocesses the content
"""

import io
import re
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
# nltk.data.find() raises OSError (not just LookupError) when the path is missing on some versions
for resource, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except (LookupError, OSError):
        nltk.download(resource, quiet=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file path.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text, or empty string on failure.
    """
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
    """
    Extract text from a Streamlit-uploaded file object.

    Args:
        file_object: File uploaded via Streamlit (has a .read() method).

    Returns:
        Extracted text, or empty string on failure.
    """
    try:
        # Read once into memory; supports both BytesIO and Streamlit UploadedFile
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
    """
    Clean and normalize text.

    Removes URLs, email addresses, and non-alphanumeric characters
    (preserving +, -, #, .) then collapses whitespace.

    Args:
        text: Raw text to preprocess.

    Returns:
        Cleaned, lowercased text.
    """
    if not text:
        return ""

    text = text.lower()
    # Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Keep alphanumerics, spaces, and a few meaningful punctuation chars
    text = re.sub(r"[^a-z0-9\s+\-#.]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(resume_text: str, skills_list: list) -> list:
    """
    Extract skills from resume text by matching against a predefined list.

    Uses word-boundary matching so "java" does not match "javascript".

    Args:
        resume_text: Resume text to analyze.
        skills_list: Skills to look for.

    Returns:
        Deduplicated list of matched skills (lowercase).
    """
    if not resume_text or not skills_list:
        return []

    resume_lower = resume_text.lower()
    found = set()

    for skill in skills_list:
        skill_lower = skill.lower()
        # Escape dots etc. so "node.js" is matched literally
        pattern = r"\b" + re.escape(skill_lower) + r"\b"
        if re.search(pattern, resume_lower):
            found.add(skill_lower)

    return list(found)


def extract_contact_info(resume_text: str) -> dict:
    """
    Extract contact information from resume text.

    Args:
        resume_text: Resume text to analyze.

    Returns:
        Dictionary with keys 'email', 'phone', and/or 'linkedin'.
    """
    contact_info: dict = {}

    if not resume_text:
        return contact_info

    # Email
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    emails = re.findall(email_pattern, resume_text)
    if emails:
        contact_info["email"] = emails[0]

    # Phone  (handles +1, country codes, dashes, dots, spaces, parentheses)
    phone_pattern = r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    phones = re.findall(phone_pattern, resume_text)
    if phones:
        contact_info["phone"] = phones[0]

    # LinkedIn
    linkedin_pattern = r"linkedin\.com/in/[\w\-]+"
    linkedin = re.findall(linkedin_pattern, resume_text, re.IGNORECASE)
    if linkedin:
        contact_info["linkedin"] = linkedin[0]

    return contact_info


def tokenize_and_filter(text: str) -> list:
    """
    Tokenize text and remove English stopwords.

    Args:
        text: Text to tokenize.

    Returns:
        List of lowercase, alphanumeric, non-stopword tokens.
    """
    if not text:
        return []

    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t.isalnum() and t not in stop_words]
