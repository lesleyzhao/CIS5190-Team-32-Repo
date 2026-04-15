from pathlib import Path
import json
import re
import time
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Current script directory + file names
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "url_only_data.csv"
OUTPUT_CSV = BASE_DIR / "scraped_headlines.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def infer_source(url: str) -> str:
    """Infer the news source from the URL."""
    url = str(url).lower()
    if "foxnews.com" in url:
        return "FoxNews"
    if "nbcnews.com" in url:
        return "NBC"
    return "Unknown"




def try_jsonld_headline(soup: BeautifulSoup):
    """Try to extract headline from JSON-LD."""
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
            candidates = data if isinstance(data, list) else [data]

            for item in candidates:
                if isinstance(item, dict):
                    if "headline" in item and isinstance(item["headline"], str):
                        return item["headline"]
                    if "@graph" in item and isinstance(item["@graph"], list):
                        for sub in item["@graph"]:
                            if isinstance(sub, dict) and "headline" in sub:
                                return sub["headline"]
        except Exception:
            continue
    return None



def try_article_date(soup: BeautifulSoup):
    """Try to extract publication date from JSON-LD."""
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
            candidates = data if isinstance(data, list) else [data]

            for item in candidates:
                if isinstance(item, dict):
                    if "datePublished" in item:
                        return item["datePublished"]
                    if "@graph" in item and isinstance(item["@graph"], list):
                        for sub in item["@graph"]:
                            if isinstance(sub, dict) and "datePublished" in sub:
                                return sub["datePublished"]
        except Exception:
            continue
    return None



def scrape_headline(url: str):
    """Scrape headline and publication date from a news article URL."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return {
                "headline_raw": None,
                "article_date": None,
                "scrape_status": "failed",
                "error_message": f"HTTP {response.status_code}",
            }

        soup = BeautifulSoup(response.text, "html.parser")

        # 1) og:title
        og = soup.find("meta", attrs={"property": "og:title"})
        title = og["content"] if og and og.get("content") else None

        # 2) twitter:title
        if not title:
            twitter = soup.find("meta", attrs={"name": "twitter:title"})
            if twitter and twitter.get("content"):
                title = twitter["content"]

        # 3) JSON-LD headline
        if not title:
            title = try_jsonld_headline(soup)

        # 4) h1
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(" ", strip=True)

        # 5) <title>
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)

        article_date = try_article_date(soup)

        if not title:
            return {
                "headline_raw": None,
                "article_date": article_date,
                "scrape_status": "failed",
                "error_message": "No headline found",
            }

        return {
            "headline_raw": title,
            "article_date": article_date,
            "scrape_status": "success",
            "error_message": None,
        }

    except Exception as error:
        return {
            "headline_raw": None,
            "article_date": None,
            "scrape_status": "failed",
            "error_message": str(error),
        }


# Read the input CSV from the current script directory
df = pd.read_csv(INPUT_CSV)

# Standardize the first column name as "url"
df = df.rename(columns={df.columns[0]: "url"})
df["source"] = df["url"].apply(infer_source)

results = []
for i, url in enumerate(df["url"]):
    result = scrape_headline(url)
    results.append(result)

    # Avoid sending requests too quickly
    time.sleep(0.5)

    if (i + 1) % 100 == 0:
        print(f"Done: {i + 1}/{len(df)}")

result_df = pd.concat([df, pd.DataFrame(results)], axis=1)


# Remove exact duplicates
result_df = result_df.drop_duplicates(subset=["url", "headline_raw"])

# Save the output CSV to the current script directory
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")
print(result_df["scrape_status"].value_counts(dropna=False))
