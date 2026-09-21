
import re


# ============================================================
# EXTRACT NAME
# ============================================================

def extract_name(text):

    # First try to find a name before Email/Phone
    header_match = re.search(
        r"^\s*([A-Za-z]+(?:\s+[A-Za-z]+){1,3})\s+"
        r"(?:Email|E-mail|Phone|Mobile)",
        text,
        re.IGNORECASE
    )

    if header_match:
        return header_match.group(1).strip()

    # Try line-by-line extraction
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines:

        lower_line = line.lower()

        if (
            "email" in lower_line
            or "phone" in lower_line
            or "mobile" in lower_line
            or "career objective" in lower_line
            or "objective" in lower_line
            or "education" in lower_line
            or "skills" in lower_line
        ):
            continue

        if lower_line in [
            "resume",
            "curriculum vitae",
            "cv",
            "profile"
        ]:
            continue

        words = line.split()

        if 1 <= len(words) <= 4:

            if all(
                re.match(
                    r"^[A-Za-z.]+$",
                    word
                )
                for word in words
            ):
                return line

    return "Not found"


# ============================================================
# EXTRACT EMAIL
# ============================================================

def extract_email(text):

    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    if match:
        return match.group(0)

    return "Not found"


# ============================================================
# EXTRACT PHONE
# ============================================================

def extract_phone(text):

    match = re.search(
        r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
        text
    )

    if match:

        phone = re.sub(
            r"\D",
            "",
            match.group(0)
        )

        if phone.startswith("91") and len(phone) == 12:
            phone = phone[2:]

        return phone

    return "Not found"


# ============================================================
# EXTRACT EDUCATION
# ============================================================

def extract_education(text):

    normalized_text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    degree_pattern = (
        r"\b("
        r"B\.?Tech"
        r"|B\.?E"
        r"|B\.?Sc"
        r"|B\.?Com"
        r"|M\.?Tech"
        r"|M\.?E"
        r"|M\.?Sc"
        r"|MCA"
        r"|MBA"
        r"|Ph\.?D"
        r")\b"
    )

    degree_match = re.search(
        degree_pattern,
        normalized_text,
        re.IGNORECASE
    )

    if degree_match:

        education_text = normalized_text[
            degree_match.start():
        ]

        stop_match = re.search(
            r"\b(?:CGPA|GPA|Skills?|Projects?|"
            r"Certifications?|Experience|"
            r"Work Experience|Achievements?)\b",
            education_text,
            re.IGNORECASE
        )

        if stop_match:

            education_text = education_text[
                :stop_match.start()
            ]

        education_text = education_text.strip(
            " :-|,"
        )

        if education_text:
            return education_text

    return "Not found"


# ============================================================
# EXTRACT CGPA
# ============================================================

def extract_cgpa(text):

    match = re.search(
        r"\b(?:CGPA|GPA)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return "Not found"


# ============================================================
# EXTRACT SKILLS
# ============================================================

def extract_skills(text):

    skills_list = [
        "java",
        "python",
        "c++",
        "c#",
        "c",
        "sql",
        "mysql",
        "mongodb",
        "git",
        "github",
        "rest api",
        "spring boot",
        "spring",
        "html",
        "css",
        "javascript",
        "react",
        "angular",
        "machine learning",
        "deep learning",
        "nlp",
        "tensorflow",
        "pytorch",
        "aws",
        "azure",
        "docker",
        "kubernetes"
    ]

    normalized_text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    skills_match = re.search(
        r"\bskills?\b\s*:?",
        normalized_text,
        re.IGNORECASE
    )

    if skills_match:

        skills_text = normalized_text[
            skills_match.end():
        ]

        next_section = re.search(
            r"\b(?:projects?|certifications?|"
            r"experience|work experience|"
            r"education|achievements?|"
            r"languages?|interests?|hobbies?)\b",
            skills_text,
            re.IGNORECASE
        )

        if next_section:

            skills_text = skills_text[
                :next_section.start()
            ]

    else:

        # If the Skills heading isn't available,
        # search the complete resume.
        skills_text = normalized_text


    skills_text_lower = skills_text.lower()

    found_skills = []


    # --------------------------------------------------------
    # Check longer/more specific skills first
    # --------------------------------------------------------

    for skill in skills_list:

        if skill == "c++":

            pattern = r"(?<!\w)c\+\+(?!\w)"

        elif skill == "c#":

            pattern = r"(?<!\w)c#(?!\w)"

        elif skill == "c":

            pattern = r"(?<!\w)c(?!\w)"

        else:

            pattern = (
                r"(?<!\w)"
                + re.escape(skill)
                + r"(?!\w)"
            )

        if re.search(
            pattern,
            skills_text_lower
        ):

            found_skills.append(skill)


    # --------------------------------------------------------
    # Remove "spring" if "spring boot" is already present
    # --------------------------------------------------------

    if "spring boot" in found_skills:

        found_skills = [
            skill
            for skill in found_skills
            if skill != "spring"
        ]


    return found_skills


# ============================================================
# EXTRACT ALL CANDIDATE INFORMATION
# ============================================================

def extract_candidate_info(text):

    return {

        "name": extract_name(text),

        "email": extract_email(text),

        "phone": extract_phone(text),

        "education": extract_education(text),

        "cgpa": extract_cgpa(text),

        "skills": extract_skills(text)

    }

