# CIS5190-Team-32-Repo

Link to our latex report: https://www.overleaf.com/project/69dee6a89637524f6583b00b

Link to leaderboard: https://huggingface.co/spaces/cis4190/NewsHeadlineClassifier
(All submissions start with "team_32" are our submissions.)

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/lesleyzhao/CIS5190-Team-32-Repo.git
cd CIS5190-Team-32-Repo
```

### 2. Set Up Virtual Environment on VS Code
```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install pandas torch nltk scikit-learn transformers
```

### 4. Run Preprocessing
Cleans the raw scraped headlines and saves a model-ready CSV:
```bash
python3 preprocess.py
```
Output: `scraped_headlines_clean_latest.csv`

### 5. Run Baseline Model
```bash
python3 tfidf_baseline.py
```

### 6. Run Improved Models
model.py contains the best model DistillBERT so far.
train.py is used for training with model.py.

Other Experimental Models:
- roberta.py and train_roberta.py
```bash
python3 model.py
python3 train.py
python3 roberta.py
python3 train_roberta.py
```

## Project Structure
```
CIS5190-Team-32-Repo/
├── url_only_data.csv                  # Input: 3,815 article URLs
├── scraped_headlines.csv              # Scraped raw headlines
├── scraped_headlines_clean_latest.csv # Cleaned, model-ready dataset
│
├── data_scraping.py                   # Web scraper (5-strategy headline extraction)
├── data_cleaning.py                   # Standalone cleaning script
├── preprocess.py                      # Full pipeline + leaderboard entry point
│
├── tfidf_baseline.py                  # Course baseline (TF-IDF + LR, 100 features)
├── baseline_experiments.py            # Classical model sweep (6 variants)
│
├── model_char_ngram_logreg.py         # Char n-gram LR wrapper (leaderboard-ready)
├── train_char_ngram_logreg.py         # Train + export char n-gram LR
│
├── model_HashingBoWClassifier.py      # Hashing BoW neural model
├── train_HashingBoWClassifier.py      # Training script for Hashing BoW
│
├── model.py                           # DistilBERT classifier + FallbackModel
├── train.py                           # Training script for DistilBERT
│
├── roberta.py                         # RoBERTa classifier
├── train_roberta.py                   # Training script for RoBERTa
│
├── eval_project_b.py                  # Local evaluator (mirrors leaderboard logic)
│
├── EXPERIMENT_SUMMARY.md              # Classical model results and takeaways
└── CHAR_NGRAM_LOGREG_SUBMISSION.md    # Submission notes for char n-gram model
```

---

## Label Convention

| Label | Class |
|---|---|
| `1` | Fox News |
| `0` | NBC News |

---

## Team

**Team 32** — CIS 4190/5190 Applied Machine Learning, Spring 2026

- Xiaoyang Jing — data collection, scraping, Hashing BoW model
- Lesley Zhao — preprocessing, DistilBERT & RoBERTa fine-tuning, leaderboard submissions
- League Wang — classical model sweep, feature analysis, report
