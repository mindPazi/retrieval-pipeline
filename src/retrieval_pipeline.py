import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import *


class Pipeline:
    def __init__(
        self,
        corpus_file,
        question_file,
        chunker,
        embed_model,
        chunk_size,
        chunk_overlap,
        level,
        k,
    ):
        self.corpus_file = corpus_file
        self.question_file = question_file
        self.chunker = chunker
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.level = level
        self.k = k

    def load_data(self, file_md):
        print("Loading data...")
        corpus = []

        questions_df = self.load_questions_and_normalize(self.question_file, file_md)
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            corpus = f.readlines()

        corpus = [self.clean_unk_tokens(doc) for doc in corpus]

        print(f"Loaded {len(corpus)} documents and {len(questions_df)} questions")
        print(f"{file_md} contains {len(corpus)} rows.")
        print(f"'questions_df.csv' contains {len(questions_df)} rows.\n")

        return corpus, questions_df

    def load_questions_and_normalize(self, question_file, file_md):
        try:
            questions_df = pd.read_csv(question_file)
        except FileNotFoundError:
            print(f"File not found: {question_file}")
            return None

        questions_df = questions_df[questions_df["corpus_id"] == file_md.split(".")[0]]
        questions_df["references"] = questions_df["references"].apply(json.loads)
        questions_df["references"] = questions_df["references"].apply(
            self.normalize_references
        )

        return questions_df

    def normalize_references(self, refs):
        normalized = []
        for item in refs:
            if isinstance(item, dict):
                normalized.append((item["start_index"], item["end_index"]))
            else:
                raise ValueError(f"Unexpected format in references: {item}")
        return normalized

    def clean_unk_tokens(self, text):
        # Replace <unk> with 5 spaces (preserves the original length)
        text = text.replace("<unk>", " " * 5)
        # Replace @ with 1 space
        text = text.replace("@", " ")
        return text

    def chunk_text(self, corpus):
        """Split the text into chunks of defined size"""
        print("Chunking the text...")
        chunked_corpus = []
        for doc in corpus:
            chunked_corpus.extend(self.chunker.split_text(doc))

        print(f"Chunked the text into {len(chunked_corpus)} chunks\n")
        return chunked_corpus

    def generate_embeddings(self, texts):
        """Generate embeddings for the given texts using a pre-trained model"""
        print("Generating embeddings...")
        model = SentenceTransformer(self.embed_model)
        embeddings = model.encode(texts, show_progress_bar=True)

        return embeddings

    def retrieve_top_k_answers(self, chunk_embeddings, question_embeddings):
        """
        For each question, retrieve top-k most similar chunks.
        Returns a list of lists of indices.
        """
        print("Retrieving answers...")
        results = []
        for q_embed in question_embeddings:
            sims = cosine_similarity([q_embed], chunk_embeddings)[0]
            top_k_indices = np.argsort(sims)[::-1][: self.k]
            results.append(top_k_indices)

        print(f"Retrieved {len(results)} answers\n")
        return results
