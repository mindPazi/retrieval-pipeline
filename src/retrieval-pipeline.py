import os
import pandas as pd
import json


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 125
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    corpus = []

    corpus_file = os.path.join(BASE_DIR, "data", "corpora", "wikitexts.md")
    question_file = os.path.join(BASE_DIR, "data", "questions_df.csv")

    questions_df = load_questions_and_normalize(question_file)
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = f.readlines()

    return corpus, questions_df


def load_questions_and_normalize(question_file):
    try:
        questions_df = pd.read_csv(question_file)
    except FileNotFoundError:
        print(f"File not found: {question_file}")
        return None

    questions_df = questions_df[questions_df["corpus_id"] == "wikitexts"]
    questions_df["references"] = questions_df["references"].apply(json.loads)
    questions_df["references"] = questions_df["references"].apply(normalize_references)

    return questions_df


def normalize_references(refs):
    normalized = []
    for item in refs:

        if isinstance(item, dict):
            normalized.append((item["start_index"], item["end_index"]))
        else:
            raise ValueError(f"Unexpected format in references: {item}")
    return normalized


def main():
    print("Loading data...")
    try:
        corpus, questions_df = load_data()
        print(f"Loaded {len(corpus)} documents and {len(questions_df)} questions")
        print(f"'wikitexts.md' contains {len(corpus)} rows.")
        print(f"'questions_df.csv' contains {len(questions_df)} rows.\n")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
