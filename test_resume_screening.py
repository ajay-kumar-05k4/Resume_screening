"""
Unit tests for Resume Screening System
Run with: pytest test_resume_screening.py -v
"""

import pytest
import importlib.util as _ilu
import os as _os

# Load local parser.py by path to avoid shadowing Python's built-in 'parser' module
_here = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("resume_parser", _os.path.join(_here, "parser.py"))
_parser_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_parser_mod)

preprocess_text    = _parser_mod.preprocess_text
extract_skills     = _parser_mod.extract_skills
extract_contact_info = _parser_mod.extract_contact_info
tokenize_and_filter  = _parser_mod.tokenize_and_filter

from scorer import calculate_ats_score, get_score_interpretation, calculate_keyword_match, rank_candidates


class TestTextProcessing:
    """Test text preprocessing functions"""
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        text = "I have Python & SQL skills!"
        result = preprocess_text(text)
        assert "python" in result
        assert "&" not in result
        assert result.islower()
    
    def test_preprocess_text_removes_special_chars(self):
        """Test removal of special characters"""
        text = "Email@123.com #hashtag"
        result = preprocess_text(text)
        assert "@" not in result
        assert "#" not in result
    
    def test_preprocess_text_removes_urls(self):
        """Test removal of URLs"""
        text = "Visit https://example.com and www.test.com"
        result = preprocess_text(text)
        assert "https" not in result
        assert "www" not in result
    
    def test_tokenize_and_filter(self):
        """Test tokenization and filtering"""
        text = "I am learning Python and SQL"
        tokens = tokenize_and_filter(text)
        assert "learning" in tokens
        assert "python" in tokens
        assert "sql" in tokens
        # Stopwords should be removed
        assert "i" not in tokens
        assert "am" not in tokens


class TestSkillExtraction:
    """Test skill extraction functions"""
    
    def test_extract_skills_basic(self):
        """Test basic skill extraction"""
        resume = "I have expertise in Python, SQL, and Java"
        skills_list = ["python", "sql", "java", "javascript"]
        found = extract_skills(resume, skills_list)
        assert "python" in found
        assert "sql" in found
        assert "java" in found
        assert "javascript" not in found
    
    def test_extract_skills_case_insensitive(self):
        """Test case-insensitive skill matching"""
        resume = "PYTHON SQL Power BI"
        skills_list = ["python", "sql", "power bi"]
        found = extract_skills(resume, skills_list)
        assert len(found) >= 2
    
    def test_extract_skills_empty(self):
        """Test extraction with no matching skills"""
        resume = "I know how to work"
        skills_list = ["python", "java", "sql"]
        found = extract_skills(resume, skills_list)
        assert len(found) == 0


class TestContactExtraction:
    """Test contact information extraction"""
    
    def test_extract_email(self):
        """Test email extraction"""
        text = "Contact: john.doe@example.com"
        contact = extract_contact_info(text)
        assert "email" in contact
        assert "john.doe@example.com" in contact["email"]
    
    def test_extract_phone(self):
        """Test phone number extraction"""
        text = "Phone: (555) 123-4567"
        contact = extract_contact_info(text)
        assert "phone" in contact
    
    def test_extract_linkedin(self):
        """Test LinkedIn extraction"""
        text = "LinkedIn: linkedin.com/in/johndoe"
        contact = extract_contact_info(text)
        assert "linkedin" in contact
    
    def test_extract_multiple_contacts(self):
        """Test extraction of multiple contact methods"""
        text = "john@example.com | (555) 123-4567 | linkedin.com/in/john"
        contact = extract_contact_info(text)
        assert len(contact) >= 2


class TestScoring:
    """Test ATS scoring functions"""
    
    def test_calculate_ats_score_range(self):
        """Test that ATS score is within 0-100"""
        resume = "Python SQL Data Analysis"
        job_desc = "Python SQL Data Analysis"
        score = calculate_ats_score(resume, job_desc)
        assert 0 <= score <= 100
    
    def test_calculate_ats_score_perfect_match(self):
        """Test score for perfect match"""
        text = "Python SQL Analysis"
        score = calculate_ats_score(text, text)
        assert score == 100
    
    def test_calculate_ats_score_no_match(self):
        """Test score for no match"""
        resume = "AAAA BBBB CCCC"
        job_desc = "XXXX YYYY ZZZZ"
        score = calculate_ats_score(resume, job_desc)
        assert score < 50


class TestScoreInterpretation:
    """Test score interpretation"""
    
    def test_excellent_match(self):
        """Test excellent match interpretation"""
        result = get_score_interpretation(95)
        assert "Excellent" in result["level"]
        assert "Interview" in result["recommendation"]
    
    def test_strong_match(self):
        """Test strong match interpretation"""
        result = get_score_interpretation(85)
        assert "Strong" in result["level"]
    
    def test_good_match(self):
        """Test good match interpretation"""
        result = get_score_interpretation(75)
        assert "Good" in result["level"]
    
    def test_poor_match(self):
        """Test poor match interpretation"""
        result = get_score_interpretation(50)
        assert "Needs Improvement" in result["level"]


class TestKeywordMatching:
    """Test keyword matching functions"""
    
    def test_keyword_match_basic(self):
        """Test basic keyword matching"""
        resume = "Python SQL Excel Power BI"
        job_desc = "Python SQL"
        result = calculate_keyword_match(resume, job_desc)
        assert result["match_percentage"] > 0
        assert result["matched_count"] > 0
    
    def test_keyword_match_percentage_range(self):
        """Test that match percentage is 0-100"""
        resume = "Some content"
        job_desc = "Different content"
        result = calculate_keyword_match(resume, job_desc)
        assert 0 <= result["match_percentage"] <= 100


class TestRanking:
    """Test candidate ranking functions"""
    
    def test_rank_candidates_ordering(self):
        """Test that candidates are ranked correctly"""
        candidates = [
            {"name": "Alice", "score": 75},
            {"name": "Bob", "score": 90},
            {"name": "Carol", "score": 85},
        ]
        ranked = rank_candidates(candidates)
        assert ranked[0]["name"] == "Bob"
        assert ranked[1]["name"] == "Carol"
        assert ranked[2]["name"] == "Alice"
    
    def test_rank_candidates_have_rank_field(self):
        """Test that ranked candidates have rank field"""
        candidates = [
            {"name": "Alice", "score": 80},
            {"name": "Bob", "score": 90},
        ]
        ranked = rank_candidates(candidates)
        assert all("rank" in c for c in ranked)
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow"""
        resume = """
        John Doe
        john@example.com | (555) 123-4567
        
        Skills: Python, SQL, Power BI, Excel, Data Analysis
        
        Experience: 5 years as Data Analyst
        """
        
        job_desc = """
        Senior Data Analyst
        Required: Python, SQL, Power BI
        Experience: 3+ years
        """
        
        # Extract contact
        contact = extract_contact_info(resume)
        assert len(contact) > 0
        
        # Extract skills
        skills = extract_skills(resume, ["python", "sql", "power bi", "excel"])
        assert len(skills) >= 3
        
        # Calculate score
        score = calculate_ats_score(resume, job_desc)
        assert score > 70
        
        # Get interpretation
        interp = get_score_interpretation(score)
        assert "level" in interp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
