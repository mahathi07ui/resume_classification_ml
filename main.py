import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def clean_text(text):
    """Basic text cleaning for resume content."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def main():
    # 1. Load dataset
    data = pd.read_csv("resume.csv")

    required_columns = {"Category", "Resume"}
    if not required_columns.issubset(data.columns):
        raise ValueError("resume.csv must contain 'Category' and 'Resume' columns.")

    data = data.dropna(subset=["Category", "Resume"]).copy()
    data["Resume"] = data["Resume"].apply(clean_text)

    if data["Category"].nunique() < 2:
        raise ValueError("The dataset must contain at least two different categories.")

    # Stratified split requires at least two examples in every category.
    category_counts = data["Category"].value_counts()
    if (category_counts < 2).any():
        raise ValueError(
            "Each category needs at least 2 resumes for a stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        data["Resume"],
        data["Category"],
        test_size=0.2,
        random_state=42,
        stratify=data["Category"]
    )

    # 2. TF-IDF + Linear SVM model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("classifier", LinearSVC())
    ])

    # 3. Train
    print("Training the resume classification model...")
    model.fit(X_train, y_train)
    print("Training completed.")

    # 4. Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # 5. Confusion matrix
    labels = sorted(data["Category"].unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    plt.figure(figsize=(11, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.title("Resume Classification - Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved as confusion_matrix.png")

    # 6. Save model
    joblib.dump(model, "resume_model.pkl")
    print("Trained model saved as resume_model.pkl")

    # 7. Predict a new resume
    print("\n===== NEW RESUME CLASSIFICATION =====")
    resume_text = input("Paste resume text (or press Enter to skip):\n")

    if resume_text.strip():
        predicted_category = model.predict([clean_text(resume_text)])[0]
        print(f"\nPredicted job category: {predicted_category}")
        print("Please review the result; this model is a decision-support demo.")
    else:
        print("Prediction skipped.")


if __name__ == "__main__":
    main()
