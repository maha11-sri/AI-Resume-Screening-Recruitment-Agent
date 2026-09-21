import re


def clean_text(text):
    if not text:
        return ""

    text = text.lower()

    # Remove email addresses
    text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', ' ', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{10}\b', ' ', text)

    # Protect decimal numbers
    text = re.sub(r'(\d+)\.(\d+)', r'\1DECIMAL\2', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Restore decimal points
    text = text.replace('DECIMAL', '.')

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()