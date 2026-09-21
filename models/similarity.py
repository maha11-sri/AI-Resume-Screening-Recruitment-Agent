
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text before calculating similarity.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Normalize common punctuation
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TF-IDF SIMILARITY
# ============================================================

def calculate_similarity(job_description, resume_text):
    """
    Calculate TF-IDF cosine similarity between
    job description and candidate resume.

    Returns:
        float:
            Similarity score between 0.0 and 1.0
    """

    job_description = normalize_text(job_description)
    resume_text = normalize_text(resume_text)

    # --------------------------------------------------------
    # Empty text protection
    # --------------------------------------------------------

    if not job_description or not resume_text:
        return 0.0

    documents = [
        job_description,
        resume_text
    ]

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    try:

        tfidf_matrix = vectorizer.fit_transform(
            documents
        )

    except ValueError:

        # Happens when there are no usable words
        return 0.0

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    score = float(
        similarity[0][0]
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    score = max(
        0.0,
        min(
            score,
            1.0
        )
    )

    return round(
        score,
        4
    )


# ============================================================
# PERCENTAGE VERSION
# ============================================================

def calculate_similarity_percentage(
    job_description,
    resume_text
):
    """
    Return TF-IDF similarity directly as a percentage.
    """

    score = calculate_similarity(
        job_description,
        resume_text
    )

    return round(
        score * 100,
        2
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
    Java, Spring Boot, REST API,
    MySQL, Git, HTML, CSS.

    Experience:
    Developed backend applications using
    Java and Spring Boot.
    Created REST APIs and worked with SQL
    databases and Git version control.
    """

    score = calculate_similarity(
        job_description,
        resume
    )

    percentage = calculate_similarity_percentage(
        job_description,
        resume
    )

    print("\n========== SIMILARITY ANALYSIS ==========")

    print(
        "Raw Similarity:",
        score
    )

    print(
        "Similarity Percentage:",
        f"{percentage:.2f}%"
    )

