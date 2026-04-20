"""
Spec Baseline: TF-IDF + Logistic Regression
Target accuracy: ~66.49%

Run:
    python tfidf_baseline.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# load the CSV 
csv_file_path = "scraped_headlines_clean_latest.csv"
news_df = pd.read_csv(csv_file_path, encoding="utf-8-sig")

# use cleaned headline column
X = news_df["headline_clean"]
y = news_df["source"]

# drop missing
mask = X.notna() & y.notna() & (X.str.strip() != "")
X, y = X[mask], y[mask]

# split (80% train, 20% test) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# convert labels to binary (FoxNews=1, NBC=0) 
y_train = y_train.apply(lambda x: 1 if x == "FoxNews" else 0)
y_test  = y_test.apply(lambda x:  1 if x == "FoxNews" else 0)

# TF-IDF features 
vectorizer    = TfidfVectorizer(stop_words="english", max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# logistic Regression 
model = LogisticRegression(max_iter=100)
model.fit(X_train_tfidf, y_train)

# predict 
y_pred = model.predict(X_test_tfidf)

# evaluate 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
