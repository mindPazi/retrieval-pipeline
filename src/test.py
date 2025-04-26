from retrieval_pipeline import Pipeline
from utils import (
    compute_token_level,
    compute_char_level,
    parsf,
    enhance_query_with_tech_terms,
)
import importlib

LEVEL_FUNCTIONS = {
    "token": compute_token_level,
    "char": compute_char_level,
}


def main():
    args = parsf()
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
        full_text = "".join(corpus)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    chunked_corpus = pipeline.chunk_text(corpus)

    # Check if embed_queries method exists, otherwise use generate_embeddings
    questions = questions_df["question"].tolist()

    try:
        # Try using embed_queries method if it exists
        question_embeddings = pipeline.embed_queries(questions)
    except AttributeError:
        # If it doesn't exist, enhance queries manually and then generate embeddings
        print(
            "Method 'embed_queries' not found. Using enhance_query_with_tech_terms and generate_embeddings instead."
        )
        enhanced_questions = [enhance_query_with_tech_terms(q) for q in questions]

        # Print some examples of enhanced queries for debugging
        print("\n=== First 5 Modified Queries ===")
        for i, (orig, enhanced) in enumerate(zip(questions, enhanced_questions)):
            if i >= 5:
                break
            print(f"Query {i+1}:")
            print(f"  Original: {orig}")
            print(f"  Modified: {enhanced}")
            print(
                f"  Added terms: {enhanced[len(orig):] if enhanced != orig else 'None'}"
            )
            print("-" * 50)

        question_embeddings = pipeline.generate_embeddings(enhanced_questions)

    chunk_embeddings = pipeline.generate_embeddings(chunked_corpus)
    print(
        f"Generated embeddings for {len(chunk_embeddings)} chunks and {len(question_embeddings)} questions\n"
    )

    results = pipeline.retrieve_top_k_answers(chunk_embeddings, question_embeddings)

    func = LEVEL_FUNCTIONS[args.level]
    precision, recall = func(questions_df, results, chunked_corpus, full_text)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
