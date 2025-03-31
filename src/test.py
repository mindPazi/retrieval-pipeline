from retrieval_pipeline import Pipeline
from utils import compute_token_level, compute_char_level, parsef
import importlib

LEVEL_FUNCTIONS = {
    "token": compute_token_level,
    "char": compute_char_level,
}


def main():
    args = parsef()
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

    chunk_embeddings = pipeline.generate_embeddings(chunked_corpus)
    question_embeddings = pipeline.generate_embeddings(
        questions_df["question"].tolist()
    )
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
