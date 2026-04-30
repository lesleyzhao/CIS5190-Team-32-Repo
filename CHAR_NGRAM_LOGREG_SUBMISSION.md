# Char n-gram Logistic Regression Submission Candidate

This bundle packages a submission-ready classical model based on the strongest classical result from the current experiment sweep.

## Files
- `model_char_ngram_logreg.py` — leaderboard-facing wrapper with `Model`, `NewsClassifier`, and `get_model()`
- `char_ngram_logreg_submission.joblib` — trained sklearn pipeline weights
- `train_char_ngram_logreg.py` — reproducible training/export script
- `char_ngram_logreg_holdout_metrics.json` — holdout metrics from the fixed split

## Holdout result (stratified 80/20 split, random_state=42)
- Accuracy: **0.7947**
- Macro-F1: **0.7915**

## Notes
- This model uses character n-gram TF-IDF (`analyzer="char_wb"`, `ngram_range=(3,5)`) with Logistic Regression.
- It is intended to work with the existing repo `preprocess.py`, which returns cleaned headline strings and integer labels.
- If the leaderboard lets you choose any model file path, upload `model_char_ngram_logreg.py` plus the weights file.
- If the leaderboard hardcodes the filename `model.py`, temporarily copy/rename `model_char_ngram_logreg.py` to `model.py` before submission.
