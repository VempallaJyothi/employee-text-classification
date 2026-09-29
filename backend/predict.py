import re
from pathlib import Path

import fasttext

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model" / "employee_classifier.bin"

# Turns the labels from train.txt into friendly names for the user
LABEL_NAMES = {
    "__label__frontend": "Frontend",
    "__label__java_backend": "Java Backend",
    "__label__python_backend": "Python Backend",
    "__label__devops": "DevOps",
    "__label__data": "Data",
}

# Load the model ONCE when this file is imported (not on every request)
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_FILE}. Run 'python train_model.py' first."
    )
model = fasttext.load_model(str(MODEL_FILE))


def clean_text(text: str) -> str:
    """Make the input look like the training data: lowercase, no punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # punctuation -> space
    text = re.sub(r"\s+", " ", text).strip()  # collapse extra spaces
    return text


def predict_category(text: str) -> dict:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text has no usable words.")

    labels, scores = model.predict(cleaned, k=1)
    label = labels[0]
    confidence = min(float(scores[0]), 1.0)  # fastText can return 1.00001

    return {
        "category": LABEL_NAMES.get(label, label),
        "confidence": round(confidence, 2),
    }


if __name__ == "__main__":
    tests = [
        "I am a Java developer with Spring Boot experience.",
        "Experienced in React, JavaScript and CSS.",
        "Worked with AWS, Docker and Kubernetes.",
        "Experienced in Python and FastAPI.",
        "Worked with SQL and Power BI.",
        "Python and SQL for data analysis",
        "Java with Docker",
        "I enjoy cooking and playing football",
    ]
    for t in tests:
        print(t)
        print("  ->", predict_category(t))
