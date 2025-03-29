from retrieval_pipeline import Pipeline
from utils import compute_token_level, compute_char_level
import argparse
import importlib

LEVEL_FUNCTIONS = {
    "token": compute_token_level,
    "char": compute_char_level,
}


def main():
    print("Loading data...")
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_file", default="../data/corpora/wikitexts.md")
    p.add_argument("--question_file", default="../data/questions_df.csv")
    p.add_argument(
        "--chunker",
        default="fixed_token_chunker.FixedTokenChunker",
        help="module.ClassName",
    )
    p.add_argument(
        "--level",
        choices=["token", "char"],
        help="Livello di valutazione: token o char",
        default="token",
    )
    p.add_argument("--chunk_size", type=int, default=400)
    p.add_argument("--chunk_overlap", type=int, default=0)
    p.add_argument("--embed_model", default="all-MiniLM-L6-v2")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    try:
        file_md = args.corpus_file.split("/")[-1]
        chunker = getattr(
            importlib.import_module(args.chunker.rsplit(".", 1)[0]),
            args.chunker.rsplit(".", 1)[1],
        )(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        pipeline = Pipeline(
            args.corpus_file,
            args.question_file,
            chunker,
            args.embed_model,
            args.chunk_size,
            args.chunk_overlap,
            args.level,
            args.k,
        )

        corpus, questions_df = pipeline.load_data(file_md)
        print(f"Loaded {len(corpus)} documents and {len(questions_df)} questions")
        print(f"{file_md} contains {len(corpus)} rows.")
        print(f"'questions_df.csv' contains {len(questions_df)} rows.\n")
        full_text = "".join(corpus)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Chunking the text...")
    chunked_corpus = pipeline.chunk_text(corpus)
    print(f"Chunked the text into {len(chunked_corpus)} chunks\n")

    print("Generating embeddings...")
    chunk_embeddings = pipeline.generate_embeddings(chunked_corpus)
    question_embeddings = pipeline.generate_embeddings(
        questions_df["question"].tolist()
    )
    print(
        f"Generated embeddings for {len(chunk_embeddings)} chunks and {len(question_embeddings)} questions\n"
    )

    print("Retrieving answers...")
    results = pipeline.retrieve_top_k_answers(chunk_embeddings, question_embeddings)
    print(f"Retrieved {len(results)} answers\n")

    print("Computing token-level scores...")
    func = LEVEL_FUNCTIONS[args.level]
    precision, recall = func(questions_df, results, chunked_corpus, full_text)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
