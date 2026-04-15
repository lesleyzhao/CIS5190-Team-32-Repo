from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "scraped_headlines.csv"
TRAIN_READY_CSV = BASE_DIR / "scraped_headlines_clean.csv"


def repair_headline(text):
    """
    Repair common mojibake / encoding issues in the scraped headlines.
    Example:
    don?€?t -> don't
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # Fix the common broken token inside words
    text = re.sub(r"(?<=\w)\?€\?(?=\w)", "'", text)

    # Fix possessives before punctuation or space
    text = re.sub(r"(?<=s)\?€\?(?=[\s,.:;!?])", "'", text)

    # Remove any remaining broken token
    text = text.replace("?€?", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_headline(text):
    """
    Create a model-ready normalized version of the headline.
    """
    text = repair_headline(text).lower()

    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    # Keep only letters, numbers, apostrophes, and spaces
    text = re.sub(r"[^a-z0-9'\s]", " ", text)

    # Remove apostrophes inside words: don't -> dont
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # Read the scraped dataset
    df = pd.read_csv(INPUT_CSV, encoding="cp1252")

    # Start cleaning
    clean = df.copy()

    # Drop fully empty columns such as "Unnamed: 6"
    empty_cols = [col for col in clean.columns if clean[col].isna().all()]
    if empty_cols:
        clean = clean.drop(columns=empty_cols)

    # Keep only successful scrapes
    clean = clean[clean["scrape_status"] == "success"].copy()

    # Repair and normalize headlines
    clean["headline_repaired"] = clean["headline_raw"].apply(repair_headline)
    clean["headline_clean"] = clean["headline_raw"].apply(normalize_headline)

    # Remove empty cleaned headlines
    clean = clean[clean["headline_clean"].str.strip() != ""].copy()

    # Add simple QA columns
    clean["headline_char_count"] = clean["headline_repaired"].str.len()
    clean["headline_word_count"] = clean["headline_clean"].str.split().str.len()


    # Deduplicate for training
    train_ready = clean.drop_duplicates(subset=["source", "headline_clean"]).copy()
    train_ready.to_csv(TRAIN_READY_CSV, index=False, encoding="utf-8-sig")


    # Print summary
    print("Cleaning finished.")
    print(f"Saved: {TRAIN_READY_CSV.name}")


if __name__ == "__main__":
    main()