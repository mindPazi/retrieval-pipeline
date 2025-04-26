#!/usr/bin/env python3
# filter_tech_terms.py
# Legge pubmed.md da basedir/data/corpora e scrive filtered_tech_terms.txt in basedir/

import os
import re

# lista di stopword comuni da escludere (tutte lowercase)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "were",
    "using",
    "into",
    "after",
    "among",
    "based",
    "additional",
    "analysis",
    "assay",
    "activity",
    "activities",
    "samples",
    "tests",
    "data",
    "method",
    "methods",
    "results",
    "analysis",
    "paper",
    "article",
    "group",
    "groups",
    # ... aggiungi altre se necessario
}

# suffissi tipici di termini scientifici
SUFFIXES = ["ase", "ene", "ide", "ine", "ol", "one", "ose", "ium", "phage", "um", "yte"]


def extract_scientific_terms(text):
    terms = set()

    # 1. termini in corsivo Markdown *termine*
    for t in re.findall(r"\*([A-Za-z0-9\-]+)\*", text):
        terms.add(t)

    # 2. termini con suffissi
    pattern_suf = r"\b([A-Za-z]+(?:" + "|".join(SUFFIXES) + r"))\b"
    for t in re.findall(pattern_suf, text, flags=re.IGNORECASE):
        terms.add(t)

    # 3. formule chimiche con almeno una cifra
    for f in re.findall(r"\b([A-Z][A-Za-z0-9]{1,})\b", text):
        if re.search(r"\d", f):
            terms.add(f)

    return terms


def is_valid_term(term: str) -> bool:
    """
    Controlli di pulizia:
      - singola parola (no spazi)
      - lunghezza minima 3
      - non stopword
      - contiene almeno una lettera
    """
    if " " in term:
        return False
    t_low = term.lower()
    if t_low in STOPWORDS:
        return False
    if len(term) < 3:
        return False
    if not re.search(r"[A-Za-z]", term):
        return False
    return True


def main():
    # risale dalla src/ alla root del progetto
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_path = os.path.join(basedir, "data", "corpora", "pubmed.md")
    output_path = os.path.join(basedir, "filtered_tech_terms.txt")

    # leggi tutto
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # estrai e filtra
    raw_terms = extract_scientific_terms(text)
    filtered = {t for t in raw_terms if is_valid_term(t)}

    # ordina case-insensitive
    sorted_terms = sorted(filtered, key=lambda s: s.lower())

    # scrivi uno per riga
    with open(output_path, "w", encoding="utf-8") as f:
        for term in sorted_terms:
            f.write(term + "\n")


if __name__ == "__main__":
    main()
