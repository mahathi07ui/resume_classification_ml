# Resume Classification Using Machine Learning

A beginner-friendly Python project that classifies resume text into job categories using:
- NLP-style text cleaning
- TF-IDF feature extraction
- Linear Support Vector Machine (LinearSVC)
- Train/test evaluation and a confusion matrix

## Project files

- `main.py` — training, evaluation, model saving, and interactive prediction
- `resume.csv` — small illustrative sample dataset
- `requirements.txt` — Python dependencies

## Run in VS Code (Windows)

1. Extract this ZIP file.
2. Open the extracted `Resume_Classification` folder in VS Code.
3. Open Terminal > New Terminal.
4. (Recommended) Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Run:

   ```powershell
   python main.py
   ```

7. When prompted, paste resume text to classify it. Press Enter on an empty line to skip prediction.

The program creates:
- `resume_model.pkl` — saved trained model
- `confusion_matrix.png` — evaluation plot

## Dataset note

`resume.csv` is a tiny synthetic demonstration dataset with example text for five categories. It is included so the project can run immediately. Its size is **not sufficient for a reliable real-world model** and the resulting accuracy is not meaningful as a measure of hiring performance.

For a proper academic project, replace it with a larger, appropriately licensed and representative resume dataset. Keep the column names exactly `Category` and `Resume`. Try to include many examples per category and check for duplicates and class imbalance.

## Important

This is an educational decision-support project. Do not use its predictions as the sole basis for hiring, rejecting, or ranking people. Resume data can contain personal information and historical bias. Obtain permission to process resumes and protect the data.
