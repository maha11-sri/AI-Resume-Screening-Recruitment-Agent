# models/skill_matcher.py

import re
from typing import List, Dict, Tuple


# ============================================================
# SKILL ALIASES
# ============================================================

SKILL_ALIASES = {

    # --------------------------------------------------------
    # JAVA / BACKEND
    # --------------------------------------------------------

    "java": "java",

    "spring": "spring",
    "springboot": "spring boot",
    "spring boot": "spring boot",
    "spring boot framework": "spring boot",

    "rest": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "rest api": "rest api",
    "rest api development": "rest api",

    # --------------------------------------------------------
    # DATABASES
    # --------------------------------------------------------

    "sql": "sql",
    "mysql": "mysql",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mongo db": "mongodb",
    "mongodb": "mongodb",
    "oracle": "oracle",
    "redis": "redis",

    # --------------------------------------------------------
    # PROGRAMMING LANGUAGES
    # --------------------------------------------------------

    "python": "python",
    "py": "python",

    "javascript": "javascript",
    "js": "javascript",

    "typescript": "typescript",
    "ts": "typescript",

    "c++": "c++",
    "cpp": "c++",

    "c#": "c#",
    "csharp": "c#",

    # --------------------------------------------------------
    # VERSION CONTROL
    # --------------------------------------------------------

    "git": "git",
    "github": "github",
    "gitlab": "gitlab",

    # --------------------------------------------------------
    # AI / ML
    # --------------------------------------------------------

    "machine learning": "machine learning",
    "ml": "machine learning",

    "deep learning": "deep learning",
    "dl": "deep learning",

    "artificial intelligence": "artificial intelligence",
    "ai": "artificial intelligence",

    "natural language processing": "nlp",
    "nlp": "nlp",

    "computer vision": "computer vision",
    "cv": "computer vision",

    # --------------------------------------------------------
    # PYTHON / ML LIBRARIES
    # --------------------------------------------------------

    "tensorflow": "tensorflow",
    "pytorch": "pytorch",

    "pandas": "pandas",
    "numpy": "numpy",

    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",

    # --------------------------------------------------------
    # BIG DATA
    # --------------------------------------------------------

    "hadoop": "hadoop",
    "spark": "spark",
    "apache spark": "spark",

    # --------------------------------------------------------
    # CLOUD
    # --------------------------------------------------------

    "aws": "aws",
    "amazon web services": "aws",

    "azure": "azure",
    "microsoft azure": "azure",

    "gcp": "gcp",
    "google cloud": "gcp",
    "google cloud platform": "gcp",

    # --------------------------------------------------------
    # WEB DEVELOPMENT
    # --------------------------------------------------------

    "html": "html",
    "html5": "html",

    "css": "css",
    "css3": "css",

    "react": "react",
    "reactjs": "react",
    "react.js": "react",

    "angular": "angular",
    "angularjs": "angular",

    "node": "node.js",
    "nodejs": "node.js",
    "node.js": "node.js",

    "express": "express.js",
    "expressjs": "express.js",
    "express.js": "express.js",

    # --------------------------------------------------------
    # DEVOPS
    # --------------------------------------------------------

    "docker": "docker",
    "kubernetes": "kubernetes",
    "jenkins": "jenkins",

    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "ci/cd": "ci/cd",

    # --------------------------------------------------------
    # OPERATING SYSTEMS
    # --------------------------------------------------------

    "linux": "linux",
    "unix": "unix",
    "windows": "windows",

    # --------------------------------------------------------
    # COMPUTER SCIENCE
    # --------------------------------------------------------

    "data structures": "data structures",
    "dsa": "data structures",

    "algorithm": "algorithms",
    "algorithms": "algorithms",

    "oops": "oops",
    "object oriented programming": "oops",
    "object oriented programming system": "oops",

    "computer networks": "computer networks",

    "operating systems": "operating systems",
    "os": "operating systems",

    "dbms": "dbms",
    "database management system": "dbms",
}


# ============================================================
# COMMON SKILLS
# ============================================================

COMMON_SKILLS = sorted(
    set(SKILL_ALIASES.keys()),
    key=len,
    reverse=True
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text for reliable skill matching."""

    if not text:
        return ""

    text = str(text).lower()

    # Normalize Unicode dashes
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Normalize common separators
    text = text.replace("•", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # Normalize common skill variations
    text = text.replace("springboot", "spring boot")
    text = text.replace("restful api", "rest api")
    text = text.replace("restful", "rest api")

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CANONICALIZE SKILL
# ============================================================

def canonicalize_skill(skill: str) -> str:
    """Convert a skill or alias into its canonical form."""

    if not skill:
        return ""

    skill = normalize_text(skill)

    return SKILL_ALIASES.get(skill, skill)


# ============================================================
# SAFE SKILL SEARCH
# ============================================================

def contains_skill(
    text: str,
    skill: str
) -> bool:
    """Check whether a skill exists in text."""

    text = normalize_text(text)
    skill = normalize_text(skill)

    if not text or not skill:
        return False

    # Special-character skills
    if skill in {
        "c++",
        "c#",
        "node.js",
        "ci/cd"
    }:
        return bool(
            re.search(
                re.escape(skill),
                text,
                flags=re.IGNORECASE
            )
        )

    # Normal word boundary matching
    pattern = (
        r"(?<!\w)"
        + re.escape(skill)
        + r"(?!\w)"
    )

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )
    )


# ============================================================
# EXTRACT SKILLS
# ============================================================

def extract_skills(
    text: str
) -> List[str]:
    """Extract canonical skills from text."""

    text = normalize_text(text)

    if not text:
        return []

    found_skills = []

    for alias in COMMON_SKILLS:

        if contains_skill(text, alias):

            canonical = canonicalize_skill(alias)

            if canonical not in found_skills:
                found_skills.append(canonical)

    # Remove parent/child duplicates.
    #
    # Example:
    # Spring Boot -> keep Spring Boot
    # Do NOT also keep Spring
    #
    # This is especially important for job descriptions.

    if "spring boot" in found_skills:
        found_skills = [
            skill
            for skill in found_skills
            if skill != "spring"
        ]

    return found_skills


# ============================================================
# EXTRACT REQUIRED SKILLS
# ============================================================

def extract_required_skills(
    job_description: str
) -> List[str]:

    return extract_skills(job_description)


# ============================================================
# RELATED SKILLS
# ============================================================

def skills_are_equivalent(
    required_skill: str,
    candidate_skill: str
) -> bool:
    """
    Determine whether a candidate skill satisfies
    a required skill.
    """

    required_skill = canonicalize_skill(required_skill)
    candidate_skill = canonicalize_skill(candidate_skill)

    if required_skill == candidate_skill:
        return True

    # --------------------------------------------------------
    # SQL FAMILY
    # --------------------------------------------------------

    sql_skills = {
        "sql",
        "mysql",
        "postgresql",
        "oracle"
    }

    if (
        required_skill == "sql"
        and candidate_skill in sql_skills
    ):
        return True

    # --------------------------------------------------------
    # MYSQL / POSTGRES / ORACLE REQUIRE SQL
    # --------------------------------------------------------

    if (
        required_skill in {
            "mysql",
            "postgresql",
            "oracle"
        }
        and candidate_skill == "sql"
    ):
        return True

    return False


# ============================================================
# MATCH CANDIDATE SKILLS
# ============================================================

def match_skills(
    candidate_text: str,
    required_skills: List[str]
) -> Tuple[List[str], List[str]]:

    candidate_text = normalize_text(candidate_text)

    candidate_skills = extract_skills(candidate_text)

    normalized_required = []

    for skill in required_skills:

        canonical = canonicalize_skill(skill)

        if (
            canonical
            and canonical not in normalized_required
        ):
            normalized_required.append(canonical)

    # Remove "spring" when "spring boot" exists
    if "spring boot" in normalized_required:
        normalized_required = [
            skill
            for skill in normalized_required
            if skill != "spring"
        ]

    matched_skills = []
    missing_skills = []

    for required_skill in normalized_required:

        matched = False

        # ----------------------------------------------------
        # Compare extracted candidate skills
        # ----------------------------------------------------

        for candidate_skill in candidate_skills:

            if skills_are_equivalent(
                required_skill,
                candidate_skill
            ):
                matched = True
                break

        # ----------------------------------------------------
        # Direct text fallback
        # ----------------------------------------------------

        if not matched:

            if contains_skill(
                candidate_text,
                required_skill
            ):
                matched = True

        if matched:

            matched_skills.append(
                required_skill
            )

        else:

            missing_skills.append(
                required_skill
            )

    return (
        matched_skills,
        missing_skills
    )


# ============================================================
# SKILL COVERAGE
# ============================================================

def calculate_skill_coverage(
    matched_skills: List[str],
    required_skills: List[str]
) -> float:

    if not required_skills:
        return 0.0

    unique_required = set()

    for skill in required_skills:

        canonical = canonicalize_skill(skill)

        if canonical:
            unique_required.add(canonical)

    unique_matched = set()

    for skill in matched_skills:

        canonical = canonicalize_skill(skill)

        if canonical:
            unique_matched.add(canonical)

    # Remove spring if spring boot is required
    if "spring boot" in unique_required:
        unique_required.discard("spring")

    if not unique_required:
        return 0.0

    matched_count = 0

    for required_skill in unique_required:

        for matched_skill in unique_matched:

            if skills_are_equivalent(
                required_skill,
                matched_skill
            ):
                matched_count += 1
                break

    coverage = (
        matched_count
        /
        len(unique_required)
    ) * 100

    return round(
        min(
            max(
                coverage,
                0.0
            ),
            100.0
        ),
        2
    )


# ============================================================
# SKILL MATCH SCORE
# ============================================================

def calculate_skill_score(
    matched_skills: List[str],
    required_skills: List[str]
) -> float:

    return calculate_skill_coverage(
        matched_skills,
        required_skills
    )


# ============================================================
# HUMAN-READABLE EXPLANATION
# ============================================================

def generate_skill_coverage_explanation(
    matched_skills: List[str],
    missing_skills: List[str],
    required_skills: List[str]
) -> str:

    unique_required = []

    for skill in required_skills:

        canonical = canonicalize_skill(skill)

        if (
            canonical
            and canonical not in unique_required
        ):
            unique_required.append(canonical)

    # Remove Spring if Spring Boot exists
    if "spring boot" in unique_required:
        unique_required = [
            skill
            for skill in unique_required
            if skill != "spring"
        ]

    total_required = len(unique_required)

    matched_unique = set()

    for skill in matched_skills:
        canonical = canonicalize_skill(skill)

        if canonical:
            matched_unique.add(canonical)

    matched_count = 0

    for required_skill in unique_required:

        for matched_skill in matched_unique:

            if skills_are_equivalent(
                required_skill,
                matched_skill
            ):
                matched_count += 1
                break

    coverage = calculate_skill_coverage(
        matched_skills,
        unique_required
    )

    if total_required == 0:

        return (
            "No required skills were identified "
            "from the job description."
        )

    if coverage >= 80:

        level = "Very strong"

    elif coverage >= 60:

        level = "Strong"

    elif coverage >= 40:

        level = "Moderate"

    elif coverage > 0:

        level = "Low"

    else:

        level = "Very low"

    explanation = (
        f"{level} skill coverage: the candidate "
        f"matches {matched_count} out of "
        f"{total_required} required skill(s), "
        f"resulting in {coverage:.2f}% skill coverage."
    )

    if matched_skills:

        explanation += (
            " Matched skills: "
            + ", ".join(matched_skills)
            + "."
        )

    if missing_skills:

        explanation += (
            " Missing skills: "
            + ", ".join(missing_skills)
            + "."
        )

    else:

        explanation += (
            " No required skills are currently missing."
        )

    return explanation


# ============================================================
# MAIN SKILL ANALYSIS
# ============================================================

def analyze_skills(
    candidate_text: str,
    job_description: str = None,
    required_skills: List[str] = None
) -> Dict:

    # --------------------------------------------------------
    # Determine required skills
    # --------------------------------------------------------

    if required_skills is None:

        if job_description:

            required_skills = (
                extract_required_skills(
                    job_description
                )
            )

        else:

            required_skills = []

    # --------------------------------------------------------
    # Normalize required skills
    # --------------------------------------------------------

    normalized_required = []

    for skill in required_skills:

        canonical = canonicalize_skill(skill)

        if (
            canonical
            and canonical not in normalized_required
        ):
            normalized_required.append(canonical)

    # Remove Spring when Spring Boot is required
    if "spring boot" in normalized_required:

        normalized_required = [
            skill
            for skill in normalized_required
            if skill != "spring"
        ]

    # --------------------------------------------------------
    # Candidate skills
    # --------------------------------------------------------

    candidate_skills = extract_skills(
        candidate_text
    )

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    matched_skills, missing_skills = (
        match_skills(
            candidate_text,
            normalized_required
        )
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    skill_coverage = calculate_skill_coverage(
        matched_skills,
        normalized_required
    )

    skill_match = calculate_skill_score(
        matched_skills,
        normalized_required
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation = (
        generate_skill_coverage_explanation(
            matched_skills,
            missing_skills,
            normalized_required
        )
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "candidate_skills":
            candidate_skills,

        "required_skills":
            normalized_required,

        "matched_skills":
            matched_skills,

        "missing_skills":
            missing_skills,

        "matched_count":
            len(matched_skills),

        "missing_count":
            len(missing_skills),

        "required_count":
            len(normalized_required),

        "skill_match":
            skill_match,

        "skill_coverage":
            skill_coverage,

        "skill_coverage_explanation":
            explanation
    }


# ============================================================
# APP.PY COMPATIBLE FUNCTION
# ============================================================

def calculate_skill_match(
    jd_text: str,
    resume_text: str
) -> Tuple[
    float,
    List[str],
    List[str],
    List[str]
]:

    analysis = analyze_skills(
        candidate_text=resume_text,
        job_description=jd_text
    )

    return (
        analysis["skill_match"],
        analysis["matched_skills"],
        analysis["missing_skills"],
        analysis["candidate_skills"]
    )


# ============================================================
# BACKWARD COMPATIBLE FUNCTION
# ============================================================

def get_skill_match(
    candidate_text: str,
    job_description: str
) -> Dict:

    return analyze_skills(
        candidate_text=candidate_text,
        job_description=job_description
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    job_description = """
    Java Developer required with Java,
    Spring Boot, REST API, SQL and Git.
    """

    resume = """
    Rahul Sharma

    Skills:
    Java, Springboot, REST,
    MySQL, Git, HTML, CSS
    """

    result = analyze_skills(
        candidate_text=resume,
        job_description=job_description
    )

    print("\n========== SKILL ANALYSIS ==========")

    print(
        "Required Skills:",
        result["required_skills"]
    )

    print(
        "Candidate Skills:",
        result["candidate_skills"]
    )

    print(
        "Matched Skills:",
        result["matched_skills"]
    )

    print(
        "Missing Skills:",
        result["missing_skills"]
    )

    print(
        "Skill Match:",
        f'{result["skill_match"]:.2f}%'
    )

    print(
        "Skill Coverage:",
        f'{result["skill_coverage"]:.2f}%'
    )

    print(
        "Explanation:",
        result["skill_coverage_explanation"]
    )