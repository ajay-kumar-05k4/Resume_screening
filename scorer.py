"""
Resume Scorer Module
Calculates ATS scores and similarity between resumes and job descriptions.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load local parser.py explicitly to avoid colliding with Python's built-in 'parser' module
import importlib.util as _ilu
import os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("resume_parser", _os.path.join(_here, "parser.py"))
_resume_parser = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_resume_parser)
preprocess_text = _resume_parser.preprocess_text


def calculate_tfidf_similarity(resume_text: str, job_description: str) -> float:
    """
    Calculate cosine similarity between resume and job description using TF-IDF.

    Args:
        resume_text: Resume content.
        job_description: Job description.

    Returns:
        Similarity score in [0, 1].
    """
    try:
        resume_cleaned = preprocess_text(resume_text)
        jd_cleaned = preprocess_text(job_description)

        # Guard: if either document is empty after cleaning, return 0
        if not resume_cleaned.strip() or not jd_cleaned.strip():
            return 0.0

        vectorizer = TfidfVectorizer(
            max_features=500,       # More features → better discrimination
            ngram_range=(1, 2),     # Unigrams + bigrams capture phrases like "machine learning"
            sublinear_tf=True,      # Dampen high-frequency terms
        )
        vectors = vectorizer.fit_transform([resume_cleaned, jd_cleaned])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(np.clip(similarity, 0.0, 1.0))
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0


def calculate_ats_score(resume_text: str, job_description: str) -> int:
    """
    Return an ATS score in [0, 100].

    Args:
        resume_text: Resume content.
        job_description: Job description.

    Returns:
        Integer ATS score.
    """
    similarity = calculate_tfidf_similarity(resume_text, job_description)
    return int(np.clip(round(similarity * 100), 0, 100))


def get_score_interpretation(ats_score: int) -> dict:
    """
    Interpret an ATS score and return a level + recommendation.

    Args:
        ats_score: Integer score in [0, 100].

    Returns:
        Dict with keys 'level', 'color', and 'recommendation'.
    """
    # Validate input to avoid silent surprises
    score = int(np.clip(ats_score, 0, 100))

    if score >= 90:
        return {
            "level": "Excellent Match",
            "color": "green",
            "recommendation": "Strong candidate – highly recommended for interview.",
        }
    if score >= 80:
        return {
            "level": "Strong Match",
            "color": "lightgreen",
            "recommendation": "Good candidate – consider scheduling an interview.",
        }
    if score >= 70:
        return {
            "level": "Good Match",
            "color": "yellow",
            "recommendation": "Moderate candidate – evaluate more carefully before deciding.",
        }
    return {
        "level": "Needs Improvement",
        "color": "red",
        "recommendation": "Weak match – significant skill or experience gaps detected.",
    }


def calculate_keyword_match(resume_text: str, job_description: str) -> dict:
    """
    Calculate keyword-level overlap between resume and job description.

    Short common words (≤2 chars) are filtered out before comparison.

    Args:
        resume_text: Resume content.
        job_description: Job description.

    Returns:
        Dict with matched/missing/extra keywords, percentages, and counts.
    """
    resume_words = {
        w for w in preprocess_text(resume_text).split() if len(w) > 2
    }
    jd_words = {
        w for w in preprocess_text(job_description).split() if len(w) > 2
    }

    matched = resume_words & jd_words
    missing = jd_words - resume_words
    extra = resume_words - jd_words

    match_pct = round(len(matched) / len(jd_words) * 100, 2) if jd_words else 0.0

    return {
        "matched_keywords": sorted(matched),
        "missing_keywords": sorted(missing),
        "extra_keywords": sorted(extra),
        "match_percentage": match_pct,
        "total_jd_keywords": len(jd_words),
        "matched_count": len(matched),
    }


def rank_candidates(candidates_data: list) -> list:
    """
    Sort candidates by score (descending) and attach a 1-based rank.

    Args:
        candidates_data: List of dicts, each containing at least 'name' and 'score'.

    Returns:
        Sorted list with 'rank' field added.
    """
    if not candidates_data:
        return []

    sorted_candidates = sorted(
        candidates_data, key=lambda x: x.get("score", 0), reverse=True
    )
    for idx, candidate in enumerate(sorted_candidates, start=1):
        candidate["rank"] = idx
    return sorted_candidates


def calculate_skill_gap(found_skills: list, required_skills: list) -> dict:
    """
    Compare found skills against required skills and return a gap analysis.

    Args:
        found_skills: Skills extracted from the resume.
        required_skills: Skills required by the job description.

    Returns:
        Dict with matched/missing/extra skills and a match percentage.
    """
    found_set = {s.lower().strip() for s in found_skills if s}
    required_set = {s.lower().strip() for s in required_skills if s}

    # Avoid division-by-zero when no required skills are provided
    if not required_set:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "extra_skills": sorted(found_set),
            "skill_match_percentage": 0.0,
            "total_required": 0,
            "matched_count": 0,
            "missing_count": 0,
        }

    matched = found_set & required_set
    missing = required_set - found_set
    extra = found_set - required_set

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "extra_skills": sorted(extra),
        "skill_match_percentage": round(len(matched) / len(required_set) * 100, 2),
        "total_required": len(required_set),
        "matched_count": len(matched),
        "missing_count": len(missing),
    }
