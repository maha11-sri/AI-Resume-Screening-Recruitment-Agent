"""
AI Recruitment Agent - Tool Layer

This module exposes the existing resume-screening functionality
as reusable tools for the Recruitment Agent.

The existing parser, skill matcher, TF-IDF similarity,
semantic similarity, and candidate information modules
are reused instead of duplicating their logic.
"""

from typing import Dict, List, Any

from parser.resume_parser import (
    read_txt,
    read_pdf,
    read_docx
)

from preprocessing.text_preprocessing import clean_text

from models.candidate_info import (
    extract_candidate_info
)

from models.skill_matcher import (
    extract_required_skills,
    analyze_skills
)

from models.similarity import (
    calculate_similarity
)

from models.semantic_similarity import (
    calculate_semantic_similarity
)


# ============================================================
# 1. RESUME PARSING TOOL
# ============================================================

def parse_resume(file) -> str:
    """
    Extract text from a PDF, DOCX, or TXT resume.

    Args:
        file: Uploaded resume file.

    Returns:
        Extracted resume text.
    """

    if file is None:
        raise ValueError("No resume file was provided.")

    file_name = getattr(file, "name", "").lower()

    try:

        if file_name.endswith(".pdf"):
            text = read_pdf(file)

        elif file_name.endswith(".docx"):
            text = read_docx(file)

        elif file_name.endswith(".txt"):
            text = read_txt(file)

        else:
            raise ValueError(
                "Unsupported resume format. "
                "Please upload PDF, DOCX, or TXT."
            )

    except Exception as error:

        raise RuntimeError(
            f"Could not parse resume: {error}"
        )

    if not text or not text.strip():

        raise ValueError(
            "The uploaded resume does not contain readable text."
        )

    return text.strip()


# ============================================================
# 2. TEXT CLEANING TOOL
# ============================================================

def clean_resume_text(resume_text: str) -> str:
    """
    Clean extracted resume text using the existing
    preprocessing module.
    """

    if not resume_text:
        return ""

    return clean_text(resume_text)


# ============================================================
# 3. CANDIDATE INFORMATION TOOL
# ============================================================

def extract_candidate_information(
    resume_text: str
) -> Dict[str, Any]:
    """
    Extract candidate information such as:

    - Name
    - Email
    - Phone
    - Education
    - CGPA
    - Skills
    """

    if not resume_text:

        return {
            "name": "Unknown",
            "email": "Not found",
            "phone": "Not found",
            "education": "Not found",
            "cgpa": "Not found",
            "skills": []
        }

    try:

        information = extract_candidate_info(
            resume_text
        )

        return information

    except Exception as error:

        return {
            "name": "Unknown",
            "email": "Not found",
            "phone": "Not found",
            "education": "Not found",
            "cgpa": "Not found",
            "skills": [],
            "error": str(error)
        }


# ============================================================
# 4. JOB DESCRIPTION SKILL TOOL
# ============================================================

def extract_job_skills(
    job_description: str
) -> List[str]:
    """
    Extract required skills from the job description.
    """

    if not job_description:

        return []

    return extract_required_skills(
        job_description
    )


# ============================================================
# 5. SKILL ANALYSIS TOOL
# ============================================================

def analyze_candidate_skills(
    resume_text: str,
    job_description: str
) -> Dict[str, Any]:
    """
    Compare candidate skills against the
    requirements of the job description.

    Returns:

    - Candidate skills
    - Required skills
    - Matched skills
    - Missing skills
    - Skill match score
    - Skill coverage
    - Explanation
    """

    if not resume_text:

        raise ValueError(
            "Resume text is empty."
        )

    if not job_description:

        raise ValueError(
            "Job description is empty."
        )

    return analyze_skills(
        candidate_text=resume_text,
        job_description=job_description
    )


# ============================================================
# 6. TF-IDF SIMILARITY TOOL
# ============================================================

def calculate_text_similarity(
    job_description: str,
    resume_text: str
) -> float:
    """
    Calculate TF-IDF cosine similarity.

    Returns:
        Score between 0 and 100.
    """

    if not job_description or not resume_text:

        return 0.0

    score = calculate_similarity(
        job_description,
        resume_text
    )

    return round(
        score * 100,
        2
    )


# ============================================================
# 7. SEMANTIC SIMILARITY TOOL
# ============================================================

def calculate_semantic_match(
    job_description: str,
    resume_text: str
) -> float:
    """
    Calculate semantic similarity using
    Sentence Transformers.

    Returns:
        Score between 0 and 100.
    """

    if not job_description or not resume_text:

        return 0.0

    score = calculate_semantic_similarity(
        job_description,
        resume_text
    )

    return round(
        score,
        2
    )


# ============================================================
# 8. COMPLETE CANDIDATE SCREENING TOOL
# ============================================================

def screen_candidate(
    resume_text: str,
    job_description: str
) -> Dict[str, Any]:
    """
    Perform complete AI-based screening of one candidate.

    Existing project scoring:

        50% Skill Match
        30% Semantic Similarity
        20% TF-IDF Similarity
    """

    if not resume_text:

        raise ValueError(
            "Resume text is empty."
        )

    if not job_description:

        raise ValueError(
            "Job description is empty."
        )

    # --------------------------------------------------------
    # Candidate information
    # --------------------------------------------------------

    candidate_info = (
        extract_candidate_information(
            resume_text
        )
    )

    # --------------------------------------------------------
    # Skill analysis
    # --------------------------------------------------------

    skill_analysis = (
        analyze_candidate_skills(
            resume_text,
            job_description
        )
    )

    skill_match = float(
        skill_analysis.get(
            "skill_match",
            0.0
        )
    )

    skill_coverage = float(
        skill_analysis.get(
            "skill_coverage",
            0.0
        )
    )

    # --------------------------------------------------------
    # TF-IDF similarity
    # --------------------------------------------------------

    text_similarity = (
        calculate_text_similarity(
            job_description,
            resume_text
        )
    )

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    semantic_similarity = (
        calculate_semantic_match(
            job_description,
            resume_text
        )
    )

    # --------------------------------------------------------
    # Existing project's weighted score
    # --------------------------------------------------------

    final_score = (
        (skill_match * 0.50)
        +
        (semantic_similarity * 0.30)
        +
        (text_similarity * 0.20)
    )

    final_score = round(
        min(
            max(
                final_score,
                0.0
            ),
            100.0
        ),
        2
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    if (
        skill_coverage >= 70
        and final_score >= 50
    ):

        recommendation = (
            "Excellent Match"
        )

    elif (
        skill_coverage >= 40
        and final_score >= 30
    ):

        recommendation = (
            "Moderate Match"
        )

    else:

        recommendation = (
            "Low Match"
        )

    # --------------------------------------------------------
    # Final candidate result
    # --------------------------------------------------------

    result = {

        "candidate_info":
            candidate_info,

        "name":
            candidate_info.get(
                "name",
                "Unknown"
            ),

        "email":
            candidate_info.get(
                "email",
                "Not found"
            ),

        "phone":
            candidate_info.get(
                "phone",
                "Not found"
            ),

        "education":
            candidate_info.get(
                "education",
                "Not found"
            ),

        "cgpa":
            candidate_info.get(
                "cgpa",
                "Not found"
            ),

        "candidate_skills":
            skill_analysis.get(
                "candidate_skills",
                []
            ),

        "required_skills":
            skill_analysis.get(
                "required_skills",
                []
            ),

        "matched_skills":
            skill_analysis.get(
                "matched_skills",
                []
            ),

        "missing_skills":
            skill_analysis.get(
                "missing_skills",
                []
            ),

        "skill_match":
            round(
                skill_match,
                2
            ),

        "skill_coverage":
            round(
                skill_coverage,
                2
            ),

        "similarity":
            round(
                text_similarity,
                2
            ),

        "semantic":
            round(
                semantic_similarity,
                2
            ),

        "final_score":
            final_score,

        "recommendation":
            recommendation,

        "skill_coverage_explanation":
            skill_analysis.get(
                "skill_coverage_explanation",
                ""
            )
    }

    return result


# ============================================================
# 9. RANK CANDIDATES TOOL
# ============================================================

def rank_candidates(
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Rank screened candidates from highest
    final score to lowest.
    """

    if not candidates:

        return []

    ranked = sorted(
        candidates,
        key=lambda candidate:
            candidate.get(
                "final_score",
                0.0
            ),
        reverse=True
    )

    # Add ranking number

    for index, candidate in enumerate(
        ranked,
        start=1
    ):

        candidate["rank"] = index

    return ranked


# ============================================================
# 10. FIND CANDIDATES BY SKILL
# ============================================================

def find_candidates_by_skill(
    candidates: List[Dict[str, Any]],
    required_skill: str
) -> List[Dict[str, Any]]:
    """
    Find candidates who have a particular
    matched skill.
    """

    if not required_skill:

        return []

    search_skill = (
        required_skill.lower().strip()
    )

    matching_candidates = []

    for candidate in candidates:

        matched_skills = [
            str(skill).lower()
            for skill in candidate.get(
                "matched_skills",
                []
            )
        ]

        if search_skill in matched_skills:

            matching_candidates.append(
                candidate
            )

    return matching_candidates


# ============================================================
# 11. EXPLAIN CANDIDATE
# ============================================================

def explain_candidate(
    candidate: Dict[str, Any]
) -> List[str]:
    """
    Generate a structured explanation for
    why a candidate received their score.
    """

    explanation = []

    final_score = candidate.get(
        "final_score",
        0.0
    )

    skill_match = candidate.get(
        "skill_match",
        0.0
    )

    skill_coverage = candidate.get(
        "skill_coverage",
        0.0
    )

    semantic = candidate.get(
        "semantic",
        0.0
    )

    similarity = candidate.get(
        "similarity",
        0.0
    )

    matched_skills = candidate.get(
        "matched_skills",
        []
    )

    missing_skills = candidate.get(
        "missing_skills",
        []
    )

    recommendation = candidate.get(
        "recommendation",
        "Unknown"
    )

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    explanation.append(
        f"Final recruitment score: "
        f"{final_score:.2f}%."
    )

    # --------------------------------------------------------
    # Skill analysis
    # --------------------------------------------------------

    explanation.append(
        f"Skill coverage is "
        f"{skill_coverage:.2f}%."
    )

    if matched_skills:

        explanation.append(
            "Matched skills: "
            +
            ", ".join(matched_skills)
            +
            "."
        )

    if missing_skills:

        explanation.append(
            "Missing required skills: "
            +
            ", ".join(missing_skills)
            +
            "."
        )

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    explanation.append(
        f"Semantic similarity: "
        f"{semantic:.2f}%."
    )

    explanation.append(
        f"Text similarity: "
        f"{similarity:.2f}%."
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    explanation.append(
        f"Current recommendation: "
        f"{recommendation}."
    )

    return explanation


# ============================================================
# 12. COMPARE CANDIDATES TOOL
# ============================================================

def compare_candidates(
    candidate_a: Dict[str, Any],
    candidate_b: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform an in-depth head-to-head comparison between two candidates.
    """
    name_a = candidate_a.get("name", "Candidate A")
    name_b = candidate_b.get("name", "Candidate B")

    score_a = float(candidate_a.get("final_score", 0.0))
    score_b = float(candidate_b.get("final_score", 0.0))

    skill_a = float(candidate_a.get("skill_match", 0.0))
    skill_b = float(candidate_b.get("skill_match", 0.0))

    coverage_a = float(candidate_a.get("skill_coverage", 0.0))
    coverage_b = float(candidate_b.get("skill_coverage", 0.0))

    semantic_a = float(candidate_a.get("semantic", 0.0))
    semantic_b = float(candidate_b.get("semantic", 0.0))

    similarity_a = float(candidate_a.get("similarity", 0.0))
    similarity_b = float(candidate_b.get("similarity", 0.0))

    matched_a = set(s.lower() for s in candidate_a.get("matched_skills", []))
    matched_b = set(s.lower() for s in candidate_b.get("matched_skills", []))

    unique_to_a = sorted(matched_a - matched_b)
    unique_to_b = sorted(matched_b - matched_a)
    common_skills = sorted(matched_a & matched_b)

    if score_a > score_b:
        verdict = f"{name_a} leads by {score_a - score_b:.2f}% in overall score."
        winner = name_a
    elif score_b > score_a:
        verdict = f"{name_b} leads by {score_b - score_a:.2f}% in overall score."
        winner = name_b
    else:
        verdict = "Both candidates are currently tied with identical final match scores."
        winner = "Tie"

    return {
        "candidate_a": name_a,
        "candidate_b": name_b,
        "score_a": score_a,
        "score_b": score_b,
        "skill_match_a": skill_a,
        "skill_match_b": skill_b,
        "skill_coverage_a": coverage_a,
        "skill_coverage_b": coverage_b,
        "semantic_a": semantic_a,
        "semantic_b": semantic_b,
        "similarity_a": similarity_a,
        "similarity_b": similarity_b,
        "unique_to_a": unique_to_a,
        "unique_to_b": unique_to_b,
        "common_skills": common_skills,
        "winner": winner,
        "verdict": verdict
    }


# ============================================================
# 13. FILTER CANDIDATES TOOL
# ============================================================

def filter_candidates(
    candidates: List[Dict[str, Any]],
    skill: str = None,
    min_score: float = None,
    max_score: float = None,
    decision: str = None,
    recommendation: str = None,
    max_results: int = None
) -> List[Dict[str, Any]]:
    """
    Filter candidates by multiple search criteria.
    """
    if not candidates:
        return []

    filtered = []

    for candidate in candidates:
        if min_score is not None:
            if float(candidate.get("final_score", 0.0)) < float(min_score):
                continue

        if max_score is not None:
            if float(candidate.get("final_score", 0.0)) > float(max_score):
                continue

        if skill and skill.strip():
            target_skill = skill.lower().strip()
            cand_skills = [
                str(s).lower() for s in candidate.get("candidate_skills", [])
            ] + [
                str(s).lower() for s in candidate.get("matched_skills", [])
            ]
            if target_skill not in cand_skills:
                continue

        if decision and decision.strip() and decision.lower() != "all":
            cand_decision = str(candidate.get("recruiter_decision", "under review")).lower()
            if decision.lower().strip() not in cand_decision:
                continue

        if recommendation and recommendation.strip() and recommendation.lower() != "all":
            cand_rec = str(candidate.get("recommendation", "")).lower()
            if recommendation.lower().strip() not in cand_rec:
                continue

        filtered.append(candidate)

    if max_results and max_results > 0:
        return filtered[:max_results]

    return filtered


# ============================================================
# 14. RECOMMEND HIRING ACTIONS TOOL
# ============================================================

def recommend_hiring_actions(
    candidate: Dict[str, Any],
    job_description: str = None
) -> Dict[str, Any]:
    """
    Generate tailored interview recommendations, assessment questions,
    and risk mitigation suggestions for a candidate.
    """
    name = candidate.get("name", "Candidate")
    matched_skills = candidate.get("matched_skills", [])
    missing_skills = candidate.get("missing_skills", [])
    final_score = float(candidate.get("final_score", 0.0))
    skill_coverage = float(candidate.get("skill_coverage", 0.0))

    interview_questions = []
    for skill in matched_skills[:3]:
        interview_questions.append(
            f"Ask about practical hands-on experience with {skill.title()} and architecture decisions in past projects."
        )

    for skill in missing_skills[:3]:
        interview_questions.append(
            f"Probe readiness and willingness to quickly ramp up on required skill: {skill.title()}."
        )

    if final_score >= 65 and skill_coverage >= 70:
        action = "Fast-track to technical interview round."
        risk_level = "Low"
    elif final_score >= 40:
        action = "Conduct focused screening on missing prerequisites before technical deep dive."
        risk_level = "Medium"
    else:
        action = "Consider holding or rejecting unless supplementary portfolio or strong project work offsets gaps."
        risk_level = "High"

    return {
        "candidate_name": name,
        "recommended_action": action,
        "risk_level": risk_level,
        "recommended_questions": interview_questions,
        "missing_skill_concerns": missing_skills
    }


# ============================================================
# 15. RECRUITMENT SUMMARY TOOL
# ============================================================

def generate_recruitment_summary(
    candidates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a high-level executive summary of all screened candidates.
    """
    if not candidates:
        return {
            "total_candidates": 0,
            "average_score": 0.0,
            "excellent_matches": 0,
            "moderate_matches": 0,
            "low_matches": 0,
            "top_candidate": None,
            "most_common_missing_skills": []
        }

    total = len(candidates)
    scores = [float(c.get("final_score", 0.0)) for c in candidates]
    avg_score = round(sum(scores) / total, 2)

    excellent = sum(1 for c in candidates if "Excellent" in c.get("recommendation", ""))
    moderate = sum(1 for c in candidates if "Moderate" in c.get("recommendation", ""))
    low = sum(1 for c in candidates if "Low" in c.get("recommendation", ""))

    ranked = sorted(candidates, key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    top_candidate = ranked[0] if ranked else None

    missing_counts: Dict[str, int] = {}
    for c in candidates:
        for s in c.get("missing_skills", []):
            s_low = s.lower()
            missing_counts[s_low] = missing_counts.get(s_low, 0) + 1

    sorted_missing = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_candidates": total,
        "average_score": avg_score,
        "excellent_matches": excellent,
        "moderate_matches": moderate,
        "low_matches": low,
        "top_candidate": top_candidate.get("name") if top_candidate else None,
        "top_candidate_score": top_candidate.get("final_score") if top_candidate else 0.0,
        "most_common_missing_skills": sorted_missing[:5]
    }


# ============================================================
# 16. TOOL SCHEMAS FOR AGENT DISCOVERY & LLM FUNCTION CALLING
# ============================================================

TOOL_SCHEMAS = [
    {
        "name": "screen_candidate",
        "description": "Screens a single candidate resume against the job description, computing scores, skills, and match level.",
        "parameters": {
            "type": "object",
            "properties": {
                "resume_text": {"type": "string", "description": "Raw resume text."},
                "job_description": {"type": "string", "description": "Job description text."}
            },
            "required": ["resume_text", "job_description"]
        }
    },
    {
        "name": "rank_candidates",
        "description": "Ranks candidates in descending order of their final recruitment match score.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {"type": "array", "description": "List of candidate screening result dictionaries."}
            },
            "required": ["candidates"]
        }
    },
    {
        "name": "compare_candidates",
        "description": "Performs head-to-head comparison between two candidates, detailing score differences and skill overlap.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_a": {"type": "object", "description": "First candidate dictionary."},
                "candidate_b": {"type": "object", "description": "Second candidate dictionary."}
            },
            "required": ["candidate_a", "candidate_b"]
        }
    },
    {
        "name": "find_candidates_by_skill",
        "description": "Filters and returns candidates possessing a specific matched skill.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {"type": "array", "description": "List of screened candidates."},
                "required_skill": {"type": "string", "description": "Target skill name (e.g. 'python', 'spring boot')."}
            },
            "required": ["candidates", "required_skill"]
        }
    },
    {
        "name": "explain_candidate",
        "description": "Produces an explainable breakdown of why a candidate received their score, highlighting strengths and missing requirements.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate": {"type": "object", "description": "Candidate screening result dictionary."}
            },
            "required": ["candidate"]
        }
    },
    {
        "name": "filter_candidates",
        "description": "Filters candidates based on minimum score, required skill, or recruiter status.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {"type": "array", "description": "List of screened candidates."},
                "skill": {"type": "string", "description": "Optional skill filter."},
                "min_score": {"type": "number", "description": "Minimum score threshold (0-100)."},
                "decision": {"type": "string", "description": "Recruiter status (Shortlisted, Under Review, Rejected)."}
            },
            "required": ["candidates"]
        }
    },
    {
        "name": "recommend_hiring_actions",
        "description": "Generates tailored interview questions and next hiring actions for a candidate.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate": {"type": "object", "description": "Candidate screening result dictionary."}
            },
            "required": ["candidate"]
        }
    },
    {
        "name": "generate_recruitment_summary",
        "description": "Computes high-level recruitment pipeline statistics and common missing skill trends.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {"type": "array", "description": "List of screened candidates."}
            },
            "required": ["candidates"]
        }
    }
]


# ============================================================
# 17. TOOL REGISTRY
# ============================================================

AGENT_TOOLS = {
    "parse_resume": parse_resume,
    "clean_resume_text": clean_resume_text,
    "extract_candidate_information": extract_candidate_information,
    "extract_job_skills": extract_job_skills,
    "analyze_candidate_skills": analyze_candidate_skills,
    "calculate_text_similarity": calculate_text_similarity,
    "calculate_semantic_match": calculate_semantic_match,
    "screen_candidate": screen_candidate,
    "rank_candidates": rank_candidates,
    "find_candidates_by_skill": find_candidates_by_skill,
    "explain_candidate": explain_candidate,
    "compare_candidates": compare_candidates,
    "filter_candidates": filter_candidates,
    "recommend_hiring_actions": recommend_hiring_actions,
    "generate_recruitment_summary": generate_recruitment_summary
}