import pandas as pd
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import *
import argparse
import importlib

LEVEL_FUNCTIONS = {
    "token": compute_token_level,
    "char": compute_char_level,
}


def load_data(corpus_file, question_file, file_md):
    corpus = []

    questions_df = load_questions_and_normalize(question_file, file_md)
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = f.readlines()

    return corpus, questions_df


def load_questions_and_normalize(question_file, file_md):
    try:
        questions_df = pd.read_csv(question_file)
    except FileNotFoundError:
        print(f"File not found: {question_file}")
        return None

    questions_df = questions_df[questions_df["corpus_id"] == file_md.split(".")[0]]
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


def clean_unk_tokens(text):
    """Remove <unk> and @ tokens from the text"""
    cleaned = re.sub(r"<unk>", "", text)
    cleaned = re.sub(r"@", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def chunk_text(chunker, corpus):
    """Split the text into chunks of defined size"""
    chunked_corpus = []
    for doc in corpus:
        chunked_corpus.extend(chunker.split_text(doc))
    return chunked_corpus


def generate_embeddings(texts, model_name):
    """Generate embeddings for the given texts using a pre-trained model"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def retrieve_top_k_answers(chunk_embeddings, question_embeddings, k=5):
    """
    For each question, retrieve top-k most similar chunks.
    Returns a list of lists of indices.
    """
    results = []
    for q_embed in question_embeddings:
        sims = cosine_similarity([q_embed], chunk_embeddings)[0]
        top_k_indices = np.argsort(sims)[::-1][:k]
        results.append(top_k_indices)
    return results


def main():
    print("Loading data...")
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_file", required=True)
    p.add_argument("--question_file", required=True)
    p.add_argument(
        "--chunker",
        default="fixed_token_chunker.FixedTokenChunker",
        help="module.ClassName",
    )
    p.add_argument(
        "--level",
        choices=["token", "char"],
        required=True,
        help="Livello di valutazione: token o char",
        default="token",
    )
    p.add_argument("--chunk_size", type=int, default=400)
    p.add_argument("--chunk_overlap", type=int, default=0)
    p.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    try:
        corpus_path, question_path, file_md = (
            args.corpus_file,
            args.question_file,
            args.corpus_file.split("/")[-1],
        )
        corpus, questions_df = load_data(corpus_path, question_path, file_md)
        print(f"Loaded {len(corpus)} documents and {len(questions_df)} questions")
        print(f"{file_md} contains {len(corpus)} rows.")
        print(f"'questions_df.csv' contains {len(questions_df)} rows.\n")
        full_text = "".join(corpus)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Chunking the text...")
    module, name = args.chunker.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    chunker = cls(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunked_corpus = chunk_text(chunker, corpus)
    print(f"Chunked the text into {len(chunked_corpus)} chunks\n")

    print("Generating embeddings...")
    chunk_embeddings = generate_embeddings(chunked_corpus, args.embed_model)
    question_embeddings = generate_embeddings(
        questions_df["question"].tolist(), args.embed_model
    )
    print(
        f"Generated embeddings for {len(chunk_embeddings)} chunks and {len(question_embeddings)} questions\n"
    )

    print("Retrieving answers...")
    results = retrieve_top_k_answers(chunk_embeddings, question_embeddings, args.k)
    print(f"Retrieved {len(results)} answers\n")

    print("Computing token-level scores...")
    func = LEVEL_FUNCTIONS[args.level]
    precision, recall = func(questions_df, results, chunked_corpus, full_text)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
