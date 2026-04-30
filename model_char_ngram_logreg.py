"""
Submission-ready classical model wrapper for Project B (News Source Classification).

Model type:
- Character n-gram TF-IDF + Logistic Regression

Expected companion weights file:
- char_ngram_logreg_submission.joblib

Design goals:
- Keep the leaderboard-facing interface simple and compatible with the repo evaluator.
- Load a pre-trained sklearn pipeline from disk.
- Return integer labels: FoxNews = 1, NBC = 0.

Usage with the local evaluator style:
    python eval_project_b.py \
        --model model_char_ngram_logreg.py \
        --preprocess preprocess.py \
        --csv url_val.csv

If the leaderboard requires a file literally named `model.py`, you can temporarily
rename/copy this file to `model.py` when preparing the submission bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

import joblib

DEFAULT_WEIGHTS = "char_ngram_logreg_submission.joblib"
LABEL_MAP = {
    "foxnews": 1,
    "nbc": 0,
}


class Model:
    """Load a trained sklearn pipeline and expose predict(batch) -> List[int]."""

    def __init__(self, weights_path: str | None = None) -> None:
        base_dir = Path(__file__).resolve().parent

        candidate = weights_path
        if candidate in (None, "", "__no_weights__.pth"):
            candidate = DEFAULT_WEIGHTS

        path = Path(candidate)
        if not path.is_absolute():
            path = base_dir / path

        if not path.exists():
            raise FileNotFoundError(
                f"Could not find weights file: {path}. "
                f"Expected companion file: {DEFAULT_WEIGHTS}"
            )

        self.pipeline = joblib.load(path)

    def eval(self) -> "Model":
        return self

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = ["" if x is None else str(x) for x in batch]
        if not texts:
            return []

        raw_preds = self.pipeline.predict(texts)
        out: List[int] = []
        for pred in raw_preds:
            key = str(pred).strip().lower()
            if key in LABEL_MAP:
                out.append(LABEL_MAP[key])
            elif isinstance(pred, (int, bool)):
                out.append(int(pred))
            else:
                raise ValueError(f"Unexpected prediction label: {pred!r}")
        return out


class NewsClassifier(Model):
    """Alias for evaluators that look for NewsClassifier instead of Model."""



def get_model() -> Model:
    return Model()


if __name__ == "__main__":
    model = get_model()
    demo = [
        "trump campaign slams border policy in latest statement",
        "nbc investigation details new healthcare fraud allegations",
        "stocks rise as investors react to inflation report",
    ]
    print(model.predict(demo))
