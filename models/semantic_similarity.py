from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load model only once
model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_semantic_similarity(job_description, resume_text):
    """
    Calculates semantic similarity between Job Description
    and Candidate Resume.

    Returns:
        Similarity percentage between 0 and 100.
    """

    # Safety checks
    if not job_description or not resume_text:
        return 0.0

    job_description = str(job_description).strip()
    resume_text = str(resume_text).strip()

    if not job_description or not resume_text:
        return 0.0

    try:

        # Create embeddings
        jd_embedding = model.encode(
            [job_description],
            normalize_embeddings=True
        )

        resume_embedding = model.encode(
            [resume_text],
            normalize_embeddings=True
        )

        # Calculate cosine similarity
        similarity = cosine_similarity(
            jd_embedding,
            resume_embedding
        )[0][0]

        # Convert to percentage
        percentage = float(similarity * 100)

        # Keep value between 0 and 100
        percentage = max(
            0.0,
            min(100.0, percentage)
        )

        return percentage

    except Exception as error:

        print(
            f"Semantic similarity error: {error}"
        )

        return 0.0