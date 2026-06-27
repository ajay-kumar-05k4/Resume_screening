"""
Resume Screening System
NLP + TF-IDF powered — clean, no extras.
"""

import streamlit as st
import pandas as pd
import importlib.util as _ilu
import os as _os
import sys as _sys
from datetime import datetime

# ── load parser.py without colliding with Python built-in 'parser' ────────────
def _load_local(name: str):
    _here = _os.path.dirname(_os.path.abspath(__file__))
    spec  = _ilu.spec_from_file_location(name, _os.path.join(_here, f"{name}.py"))
    mod   = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_parser_mod            = _load_local("parser")
extract_text_from_file = _parser_mod.extract_text_from_file
extract_skills         = _parser_mod.extract_skills
extract_contact_info   = _parser_mod.extract_contact_info

if "parser" not in _sys.modules or not hasattr(_sys.modules["parser"], "extract_text_from_file"):
    _sys.modules["parser"] = _parser_mod

from scorer import (
    calculate_ats_score,
    get_score_interpretation,
    calculate_keyword_match,
    calculate_skill_gap,
)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Red Hat–style CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;600;700;900&family=Red+Hat+Text:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

[data-testid="stAppViewContainer"] {
    background: #f0f0f0;
    font-family: 'Red Hat Text', 'Overpass', sans-serif;
}
[data-testid="stHeader"],
[data-testid="stSidebar"],
#MainMenu, footer,
[data-testid="stToolbar"]  { display: none !important; visibility: hidden; }
.block-container           { padding: 0 !important; max-width: 100% !important; }

/* ─── TOP NAV ─── */
.rh-nav {
    background: #cc0000;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem;
    height: 64px;
    border-bottom: 2px solid #a30000;
}
.rh-nav-logo {
    display: flex;
    align-items: center;
    gap: .75rem;
}
.rh-nav-logo svg { flex-shrink: 0; }
.rh-nav-title {
    color: #fff;
    font-family: 'Red Hat Display', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: .02em;
}
.rh-nav-tag {
    color: #ffcdd2;
    font-size: .8rem;
    font-weight: 400;
    margin-left: .5rem;
    padding-left: .5rem;
    border-left: 1px solid rgba(255,255,255,.35);
}

/* ─── HERO BAND ─── */
.rh-hero {
    background: #21252b;
    padding: 2.2rem 2.5rem 2rem;
    border-bottom: 3px solid #cc0000;
}
.rh-hero h1 {
    font-family: 'Red Hat Display', sans-serif;
    font-size: 1.9rem;
    font-weight: 900;
    color: #fff;
    line-height: 1.2;
    margin-bottom: .45rem;
}
.rh-hero p {
    color: #b8bec7;
    font-size: .95rem;
    max-width: 680px;
    line-height: 1.55;
}

/* ─── MAIN CONTENT WRAPPER ─── */
.rh-body {
    max-width: 1180px;
    margin: 0 auto;
    padding: 2rem 2rem 3rem;
}

/* ─── SECTION LABEL ─── */
.rh-section-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #6a6e73;
    margin-bottom: .55rem;
}

/* ─── CARD ─── */
.rh-card {
    background: #fff;
    border: 1px solid #c8c8c8;
    border-radius: 2px;
    margin-bottom: 1.25rem;
    overflow: hidden;
}
.rh-card-hd {
    background: #fff;
    border-bottom: 3px solid #cc0000;
    padding: .75rem 1.4rem;
    font-family: 'Red Hat Display', sans-serif;
    font-size: .8rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #151515;
}
.rh-card-body { padding: 1.2rem 1.4rem; }

/* ─── SCORE BANNER ─── */
.rh-score {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.25rem 1.6rem;
    border-left: 6px solid #cc0000;
    background: #fff;
    border: 1px solid #c8c8c8;
    border-left-width: 6px;
    margin-bottom: 1.25rem;
}
.rh-score-num  { font-family: 'Red Hat Display', sans-serif; font-size: 3rem; font-weight: 900; line-height: 1; color: #cc0000; }
.rh-score-lbl  { font-size: 1.1rem; font-weight: 700; color: #151515; margin-bottom: .2rem; }
.rh-score-rec  { font-size: .88rem; color: #3c3f42; }

.score-excellent { border-left-color: #1e8a44 !important; }
.score-excellent .rh-score-num { color: #1e8a44; }
.score-strong    { border-left-color: #0066cc !important; }
.score-strong    .rh-score-num { color: #0066cc; }
.score-good      { border-left-color: #ec7a08 !important; }
.score-good      .rh-score-num { color: #ec7a08; }

/* ─── STAT TILES ─── */
.rh-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #c8c8c8;
    border: 1px solid #c8c8c8;
    margin-bottom: 1.25rem;
}
.rh-stat {
    background: #fff;
    text-align: center;
    padding: 1rem;
}
.rh-stat-val  { font-family: 'Red Hat Display', sans-serif; font-size: 1.9rem; font-weight: 900; color: #151515; }
.rh-stat-name { font-size: .72rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: #6a6e73; margin-top: .25rem; }

/* ─── TAGS ─── */
.rh-tags { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .5rem; }
.rh-tag  { font-size: .75rem; font-weight: 600; padding: .22rem .6rem; border: 1px solid; border-radius: 2px; }
.tag-green { border-color: #1e8a44; color: #1e8a44; background: #f3fbf5; }
.tag-red   { border-color: #cc0000; color: #cc0000; background: #fff5f5; }

/* ─── TABLE ─── */
.rh-table { width: 100%; border-collapse: collapse; font-size: .875rem; }
.rh-table thead tr { background: #21252b; }
.rh-table th {
    color: #fff; text-align: left; padding: .65rem 1rem;
    font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
}
.rh-table td { padding: .6rem 1rem; border-bottom: 1px solid #ececec; color: #333; }
.rh-table tbody tr:hover td { background: #fafafa; }
.rh-table tbody tr:first-child td { background: #fff8e1; }

/* ─── GAP TABLE ─── */
.rh-gap-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
.rh-gap-col-title { font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #6a6e73; margin-bottom: .5rem; }

/* ─── PLACEHOLDER BOX ─── */
.rh-placeholder {
    background: #fff;
    border: 1px dashed #b8b8b8;
    padding: 2.5rem 2rem;
    text-align: center;
    color: #6a6e73;
    font-size: .92rem;
    margin-top: .5rem;
}
.rh-placeholder strong { color: #cc0000; display: block; font-size: 1rem; margin-bottom: .4rem; }

/* ─── FOOTER ─── */
.rh-footer {
    background: #21252b;
    color: #6a6e73;
    text-align: center;
    font-size: .78rem;
    padding: 1rem;
    margin-top: 3rem;
    border-top: 3px solid #cc0000;
}

/* ─── Streamlit widget tweaks ─── */
.stTextArea textarea {
    border: 1px solid #c8c8c8 !important;
    border-radius: 2px !important;
    font-size: .9rem !important;
}
.stFileUploader {
    border: 1px solid #c8c8c8 !important;
    border-radius: 2px !important;
    padding: .5rem !important;
}
button[kind="primary"] {
    background: #cc0000 !important;
    border: none !important;
    border-radius: 2px !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
}
button[kind="primary"]:hover { background: #a30000 !important; }
</style>
""", unsafe_allow_html=True)

# ── NAV ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rh-nav">
  <div class="rh-nav-logo">
    <svg width="38" height="26" viewBox="0 0 38 26" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M19 0C8.507 0 0 5.82 0 13s8.507 13 19 13 19-5.82 19-13S29.493 0 19 0z" fill="#fff"/>
      <path d="M19 2.6C9.946 2.6 2.6 7.358 2.6 13S9.946 23.4 19 23.4 35.4 18.642 35.4 13 28.054 2.6 19 2.6z" fill="#cc0000"/>
      <path d="M22.75 8.45h-7.5v9.1h2.6v-3.25h2.6l2.275 3.25H25.35l-2.6-3.575c1.3-.455 2.275-1.56 2.275-2.99 0-1.755-1.43-2.535-2.275-2.535zm-.325 3.9h-4.55V11.05h4.55c.65 0 1.04.39 1.04.975 0 .585-.39.325-1.04.325z" fill="#fff"/>
    </svg>
    <span class="rh-nav-title">Resume Screening
      <span class="rh-nav-tag">NLP · TF-IDF · scikit-learn</span>
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rh-hero">
  <h1>Resume Screening System</h1>
  <p>Upload a PDF resume and paste a job description to get an instant ATS match score,
  keyword analysis, skill detection, and a downloadable report — all powered by TF-IDF and NLP.</p>
</div>
""", unsafe_allow_html=True)

# ── SKILLS MASTER LIST ─────────────────────────────────────────────────────────
DEFAULT_SKILLS = [
    "python", "java", "javascript", "sql", "react", "angular",
    "node.js", "django", "flask", "spring", "kubernetes", "docker",
    "aws", "azure", "gcp", "git", "ci/cd", "linux",
    "data analysis", "power bi", "tableau", "excel", "pandas",
    "machine learning", "tensorflow", "pytorch", "nlp", "computer vision",
    "agile", "scrum", "rest api", "graphql", "mongodb", "postgresql", "mysql",
    "html", "css", "webpack", "npm", "jira", "problem solving",
]

# ── BODY WRAPPER open ─────────────────────────────────────────────────────────
st.markdown('<div class="rh-body">', unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single Resume", "Batch Processing"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Resume
# ════════════════════════════════════════════════════════════════════════════════
with tab_single:

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="rh-section-label">Resume (PDF)</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="single_upload",
            label_visibility="collapsed",
        )

    with col_r:
        st.markdown('<div class="rh-section-label">Job Description</div>', unsafe_allow_html=True)
        job_description = st.text_area(
            "Job description",
            height=180,
            placeholder="Paste the full job description here…",
            label_visibility="collapsed",
            key="single_jd",
        )

    if uploaded_file and job_description:
        with st.spinner("Analysing resume…"):
            resume_text = extract_text_from_file(uploaded_file)

        if not resume_text:
            st.error("Could not extract text from this PDF. Please try another file.")
        else:
            ats_score    = calculate_ats_score(resume_text, job_description)
            score_info   = get_score_interpretation(ats_score)
            kw_analysis  = calculate_keyword_match(resume_text, job_description)
            found_skills = extract_skills(resume_text, DEFAULT_SKILLS)
            contact_info = extract_contact_info(resume_text)
            required_skills = [s for s in DEFAULT_SKILLS if s.lower() in job_description.lower()]
            skill_gap    = calculate_skill_gap(found_skills, required_skills)

            # ── Score banner ──────────────────────────────────────────────────
            score_cls = {
                "Excellent Match":   "score-excellent",
                "Strong Match":      "score-strong",
                "Good Match":        "score-good",
            }.get(score_info["level"], "")

            st.markdown(f"""
            <div class="rh-score {score_cls}">
              <div class="rh-score-num">{ats_score}%</div>
              <div>
                <div class="rh-score-lbl">{score_info['level']}</div>
                <div class="rh-score-rec">{score_info['recommendation']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Stat tiles ────────────────────────────────────────────────────
            st.markdown(f"""
            <div class="rh-stats">
              <div class="rh-stat"><div class="rh-stat-val">{kw_analysis['matched_count']}</div><div class="rh-stat-name">Keywords Matched</div></div>
              <div class="rh-stat"><div class="rh-stat-val">{len(kw_analysis['missing_keywords'])}</div><div class="rh-stat-name">Keywords Missing</div></div>
              <div class="rh-stat"><div class="rh-stat-val">{len(found_skills)}</div><div class="rh-stat-name">Skills Detected</div></div>
              <div class="rh-stat"><div class="rh-stat-val">{skill_gap['matched_count']}/{skill_gap['total_required']}</div><div class="rh-stat-name">Skill Match</div></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Detail grid ───────────────────────────────────────────────────
            g1, g2 = st.columns(2, gap="large")

            with g1:
                # Contact info
                st.markdown('<div class="rh-card"><div class="rh-card-hd">Contact Information</div><div class="rh-card-body">', unsafe_allow_html=True)
                if contact_info:
                    for k, v in contact_info.items():
                        st.markdown(f"**{k.capitalize()}:** {v}")
                else:
                    st.markdown("<span style='color:#6a6e73;font-size:.88rem'>No contact information detected.</span>", unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

                # Matched keywords
                matched_tags = " ".join(
                    f'<span class="rh-tag tag-green">{w}</span>'
                    for w in sorted(kw_analysis["matched_keywords"])[:30]
                ) or "<span style='color:#6a6e73;font-size:.88rem'>None</span>"
                st.markdown(
                    f'<div class="rh-card"><div class="rh-card-hd">Matched Keywords</div>'
                    f'<div class="rh-card-body"><div class="rh-tags">{matched_tags}</div></div></div>',
                    unsafe_allow_html=True,
                )

            with g2:
                # Skills detected
                skill_tags = " ".join(
                    f'<span class="rh-tag tag-green">{s.title()}</span>'
                    for s in sorted(found_skills)
                ) or "<span style='color:#6a6e73;font-size:.88rem'>None detected</span>"
                st.markdown(
                    f'<div class="rh-card"><div class="rh-card-hd">Skills Detected</div>'
                    f'<div class="rh-card-body"><div class="rh-tags">{skill_tags}</div></div></div>',
                    unsafe_allow_html=True,
                )

                # Missing keywords
                missing_tags = " ".join(
                    f'<span class="rh-tag tag-red">{w}</span>'
                    for w in sorted(kw_analysis["missing_keywords"])[:30]
                ) or "<span style='color:#6a6e73;font-size:.88rem'>None — great coverage!</span>"
                st.markdown(
                    f'<div class="rh-card"><div class="rh-card-hd">Missing Keywords</div>'
                    f'<div class="rh-card-body"><div class="rh-tags">{missing_tags}</div></div></div>',
                    unsafe_allow_html=True,
                )

            # ── Skill Gap Analysis ────────────────────────────────────────────
            if required_skills:
                gap_matched = " ".join(
                    f'<span class="rh-tag tag-green">{s.title()}</span>'
                    for s in sorted(skill_gap["matched_skills"])
                ) or "<span style='color:#6a6e73;font-size:.88rem'>None</span>"
                gap_missing = " ".join(
                    f'<span class="rh-tag tag-red">{s.title()}</span>'
                    for s in sorted(skill_gap["missing_skills"])
                ) or "<span style='color:#1e8a44;font-size:.88rem'>All required skills found!</span>"

                st.markdown(f"""
                <div class="rh-card">
                  <div class="rh-card-hd">Skill Gap Analysis
                    <span style="font-weight:400;font-size:.78rem;color:#b0b0b0;margin-left:.5rem">({skill_gap['skill_match_percentage']}% match)</span>
                  </div>
                  <div class="rh-card-body">
                    <div class="rh-gap-grid">
                      <div>
                        <div class="rh-gap-col-title">Matched</div>
                        <div class="rh-tags">{gap_matched}</div>
                      </div>
                      <div>
                        <div class="rh-gap-col-title">Missing</div>
                        <div class="rh-tags">{gap_missing}</div>
                      </div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Download report ───────────────────────────────────────────────
            report_lines = [
                "RESUME SCREENING REPORT",
                f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"ATS SCORE : {ats_score}%",
                f"Level     : {score_info['level']}",
                f"Summary   : {score_info['recommendation']}",
                "",
                "CONTACT INFORMATION",
                *(f"  {k.capitalize()}: {v}" for k, v in contact_info.items()),
                "",
                "SKILLS FOUND",
                f"  {', '.join(sorted(found_skills)) or 'None detected'}",
                "",
                "KEYWORD MATCHING",
                f"  Matched  : {kw_analysis['matched_count']} / {kw_analysis['total_jd_keywords']} ({kw_analysis['match_percentage']}%)",
                f"  Missing  : {', '.join(sorted(kw_analysis['missing_keywords'])[:15]) or 'None'}",
                "",
                "SKILL GAP",
                f"  Match    : {skill_gap['skill_match_percentage']}%",
                f"  Missing  : {', '.join(sorted(skill_gap['missing_skills'])) or 'None'}",
            ]
            st.download_button(
                label="Download Report (.txt)",
                data="\n".join(report_lines),
                file_name=f"resume_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    else:
        st.markdown("""
        <div class="rh-placeholder">
          <strong>Ready to screen</strong>
          Upload a PDF resume and paste a job description above, then results appear automatically.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Processing
# ════════════════════════════════════════════════════════════════════════════════
with tab_batch:

    st.markdown('<div class="rh-section-label">Job Description (applied to all resumes)</div>', unsafe_allow_html=True)
    batch_jd = st.text_area(
        "Batch JD",
        height=140,
        placeholder="Paste the job description here…",
        label_visibility="collapsed",
        key="batch_jd",
    )

    st.markdown('<div style="margin-top:1rem"><div class="rh-section-label">Upload Resumes (PDF · multiple allowed)</div></div>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Upload resumes",
        type=["pdf"],
        accept_multiple_files=True,
        key="batch_upload",
        label_visibility="collapsed",
    )

    if batch_files and batch_jd:
        if st.button("Run Batch Analysis", type="primary"):
            progress = st.progress(0)
            rows = []
            for i, f in enumerate(batch_files):
                progress.progress((i + 1) / len(batch_files))
                text = extract_text_from_file(f)
                if text:
                    score   = calculate_ats_score(text, batch_jd)
                    skills  = extract_skills(text, DEFAULT_SKILLS)
                    contact = extract_contact_info(text)
                    rows.append({
                        "Rank":         0,
                        "File":         f.name,
                        "ATS Score":    f"{score}%",
                        "Level":        get_score_interpretation(score)["level"],
                        "Skills Found": len(skills),
                        "Email":        contact.get("email", "—"),
                        "Phone":        contact.get("phone", "—"),
                        "_score":       score,
                    })

            if rows:
                rows.sort(key=lambda r: r["_score"], reverse=True)
                for i, r in enumerate(rows):
                    r["Rank"] = i + 1
                    del r["_score"]

                header = "<tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr>"
                body   = "".join(
                    "<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>"
                    for r in rows
                )
                st.markdown(
                    f'<div class="rh-card">'
                    f'<div class="rh-card-hd">Results — {len(rows)} resume(s) analysed</div>'
                    f'<div class="rh-card-body" style="padding:0">'
                    f'<table class="rh-table"><thead>{header}</thead><tbody>{body}</tbody></table>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                df = pd.DataFrame(rows)
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False),
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No text could be extracted from the uploaded files.")
    else:
        st.markdown("""
        <div class="rh-placeholder">
          <strong>Batch mode</strong>
          Upload multiple PDF resumes and enter a job description above, then click Run.
        </div>
        """, unsafe_allow_html=True)


# ── BODY WRAPPER close ─────────────────────────────────────────────────────────
st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rh-footer">
  Resume Screening System &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; NLTK &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; pdfplumber
</div>
""", unsafe_allow_html=True)
