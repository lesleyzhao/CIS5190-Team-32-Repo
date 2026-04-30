"""
Train + export a submission-ready character n-gram Logistic Regression model.

Inputs:
- scraped_headlines_clean_latest.csv

Outputs:
- char_ngram_logreg_submission.joblib
- char_ngram_logreg_holdout_metrics.json

This script:
1) runs a fixed stratified 80/20 holdout evaluation for reproducible reporting
2) prints accuracy / macro-F1 / classification report
3) trains one final model on all available cleaned data
4) exports the final sklearn pipeline as a joblib file for submission use
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

CSV_PATH = Path("scraped_headlines_clean_latest.csv")
ALT_CSV_PATH = Path("/mnt/data/projzip/5190 project/scraped_headlines_clean_latest.csv")
WEIGHTS_OUT = Path("char_ngram_logreg_submission.joblib")
METRICS_OUT = Path("char_ngram_logreg_holdout_metrics.json")
SEED = 42
TEST_SIZE = 0.20


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "vec",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=SEED,
                    C=2.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )



def main() -> None:
    csv_path = CSV_PATH if CSV_PATH.exists() else ALT_CSV_PATH
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    X = df["headline_clean"].astype(str)
    y = df["source"].astype(str)

    mask = X.notna() & y.notna() & (X.str.strip() != "")
    X = X[mask]
    y = y[mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    holdout_model = build_pipeline()
    holdout_model.fit(X_train, y_train)
    y_pred = holdout_model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    report = classification_report(y_val, y_pred, output_dict=True)

    metrics = {
        "model": "char_wb_tfidf_logreg",
        "seed": SEED,
        "test_size": TEST_SIZE,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n_rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "classification_report": report,
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))

    print(f"CSV used         : {csv_path}")
    print(f"Holdout accuracy : {acc:.4f}")
    print(f"Holdout macro-F1 : {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    final_model = build_pipeline()
    final_model.fit(X, y)
    joblib.dump(final_model, WEIGHTS_OUT)

    print(f"\nSaved submission weights to: {WEIGHTS_OUT}")
    print(f"Saved holdout metrics to: {METRICS_OUT}")


if __name__ == "__main__":
    main()
