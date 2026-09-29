from pathlib import Path

import fasttext

# Build paths relative to this file, so the script works from any folder
BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "data" / "train.txt"
MODEL_FILE = BASE_DIR / "model" / "employee_classifier.bin"


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    print("Training model...")
    model = fasttext.train_supervised(
        input=str(TRAIN_FILE),
        epoch=100,      # passes over the whole dataset
        lr=0.5,         # learning rate: how big each learning step is
        wordNgrams=2,   # also learn word pairs like "spring boot"
        dim=50,         # size of each word vector
        minCount=1,     # keep every word, even if it appears once
        bucket=10000,   # slots for word pairs (default is 2 million, far too big)
        verbose=1,
    )

    MODEL_FILE.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_FILE))
    print(f"\nModel saved to: {MODEL_FILE}")

    # Accuracy on the TRAINING data (a sanity check, not a real test)
    samples, precision, recall = model.test(str(TRAIN_FILE))
    print(f"Examples tested: {samples}")
    print(f"Accuracy on training data: {precision:.2f}")

    print("\nLabels the model knows:")
    for label in model.labels:
        print(" ", label)

    # Try a few sentences the model has never seen
    tests = [
        "I am a Java developer with Spring Boot experience",
        "Experienced in React, JavaScript and CSS",
        "Worked with AWS, Docker and Kubernetes",
        "Experienced in Python and FastAPI",
        "Worked with SQL and Power BI",
    ]
    print("\nQuick predictions:")
    for text in tests:
        clean = text.lower().replace(",", "")
        labels, scores = model.predict(clean)
        print(f"  {text}\n    -> {labels[0]} ({scores[0]:.2f})")


if __name__ == "__main__":
    main()
