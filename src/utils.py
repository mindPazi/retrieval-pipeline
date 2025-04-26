import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from transformers import AutoTokenizer
import argparse


def find_query_despite_whitespace(document, query):

    normalized_query = re.sub(r"\s+", " ", query).strip()

    pattern = r"\s*".join(re.escape(word) for word in normalized_query.split())

    regex = re.compile(pattern, re.IGNORECASE)
    match_text = regex.search(document)

    if match_text:
        return (
            document[match_text.start() : match_text.end()],
            match_text.start(),
            match_text.end(),
        )
    else:
        return None


def rigorous_document_search(document: str, target: str):
    """
    This function performs a rigorous search of a target string within a document.
    It handles issues related to whitespace, changes in grammar, and other minor text alterations.
    The function first checks for an exact match of the target in the document.
    If no exact match is found, it performs a raw search that accounts for variations in whitespace.
    If the raw search also fails, it splits the document into sentences and uses fuzzy matching
    to find the sentence that best matches the target.

    Args:
        document (str): The document in which to search for the target.
        target (str): The string to search for within the document.

    Returns:
        tuple: A tuple containing the best match found in the document, its start index, and its end index.
        If no match is found, returns None.
    """
    if target.endswith("."):
        target = target[:-1]

    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    sentences = re.split(r"[.!?]\s*|\n", document)

    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if best_match[1] < 98:
        return None

    reference = best_match[0]

    start_index = document.find(reference)
    end_index = start_index + len(reference)

    return reference, start_index, end_index


def union_ranges(ranges):
    """
    Merge overlapping or contiguous ranges.

    Args:
        ranges (list of tuples): List of (start, end) tuples representing ranges.

    Returns:
        list of tuples: Merged ranges.
    """
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged_ranges = [sorted_ranges[0]]
    for current in sorted_ranges[1:]:
        prev_start, prev_end = merged_ranges[-1]
        curr_start, curr_end = current
        if curr_start <= prev_end:
            merged_ranges[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged_ranges.append(current)
    return merged_ranges


def sum_of_ranges(ranges):
    """
    Calculate the sum of lengths of a list of ranges.

    Args:
        ranges (list of tuples): List of (start, end) tuples representing ranges.

    Returns:
        int: Total length of all ranges.
    """
    return sum(end - start for start, end in ranges)


def intersect_two_ranges(range1, range2):

    start1, end1 = range1
    start2, end2 = range2

    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)
    if intersect_start <= intersect_end:
        return (intersect_start, intersect_end)
    else:
        return None


def char_to_token_span(start_char, end_char, offsets):
    start_token = len(offsets)

    for i, (s, e) in enumerate(offsets):

        if e > start_char:

            start_token = i
            break

    end_token = len(offsets)

    for i, (s, e) in enumerate(offsets):

        if s >= end_char:

            end_token = i
            break

    return (start_token, end_token)


def compute_token_level(
    questions_df,
    results,
    chunked_corpus,
    full_text,
    tokenizer_name="distilbert-base-uncased",
):
    """
    Calculate token-level Precision e Recall instead di char-level.
    """
    print("Computing token-level scores...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    tokenizer.model_max_length = 10**6

    encoding = tokenizer(full_text, return_offsets_mapping=True, truncation=False)

    offsets = encoding["offset_mapping"]

    all_precisions, all_recalls = [], []

    for i, ref_list in enumerate(questions_df["references"]):

        ref_ranges = union_ranges(ref_list)

        ref_token_ranges = []

        for s, e in ref_ranges:
            token_span = char_to_token_span(s, e, offsets)
            ref_token_ranges.append(token_span)

        pred_idxs = results[i]

        pred_token_ranges = []

        for idx in pred_idxs:

            match_text = rigorous_document_search(full_text, chunked_corpus[idx])
            if match_text:

                _, s, e = match_text

                pred_token_ranges.append(char_to_token_span(s, e, offsets))

        pred_token_ranges = union_ranges(pred_token_ranges)

        intersections = []

        for r_ref in ref_token_ranges:
            for r_pred in pred_token_ranges:
                overlap = intersect_two_ranges(r_ref, r_pred)
                if overlap is not None:
                    intersections.append(overlap)

        intersect = union_ranges(intersections)

        t_r = sum_of_ranges(pred_token_ranges)
        t_e = sum_of_ranges(ref_token_ranges)
        t_e_and_t_r = sum_of_ranges(intersect)

        precision = t_e_and_t_r / t_r if t_r else 0
        recall = t_e_and_t_r / t_e if t_e else 0

        all_precisions.append(precision)
        all_recalls.append(recall)

    return sum(all_precisions) / len(all_precisions) if all_precisions else 0.0, sum(
        all_recalls
    ) / len(all_recalls if all_recalls else 0.0)


def compute_char_level(questions_df, results, chunked_corpus, full_text):
    """
    Calculate and print char-level Precision and Recall
    """
    print("Computing char-level scores...")

    all_precisions = []
    all_recalls = []

    for i in range(len(questions_df["references"])):

        reference_spans = questions_df["references"].iloc[i]

        reference_ranges = union_ranges(reference_spans)

        pred_idxs = results[i]

        predicted_ranges = []

        for idx in pred_idxs:

            chunk_text = chunked_corpus[idx]

            match_text = rigorous_document_search(full_text, chunk_text)
            if match_text:

                _, start, end = match_text
                predicted_ranges.append((start, end))

        predicted_ranges = union_ranges(predicted_ranges)

        intersection_ranges = []

        for r_ref in reference_ranges:
            for r_pred in predicted_ranges:
                intersect = intersect_two_ranges(r_ref, r_pred)
                if intersect:
                    intersection_ranges.append(intersect)

        intersection_ranges = union_ranges(intersection_ranges)

        t_r = sum_of_ranges(predicted_ranges)
        t_e = sum_of_ranges(reference_ranges)
        t_e_and_t_r = sum_of_ranges(intersection_ranges)

        precision = t_e_and_t_r / t_r if t_r > 0 else 0
        recall = t_e_and_t_r / t_e if t_e > 0 else 0

        all_precisions.append(precision)
        all_recalls.append(recall)

    avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0

    return avg_precision, avg_recall


def enhance_query_with_tech_terms(query):
    """
    Enhances a query by adding technical terms from filtered_tech_terms.txt that appear in the query.
    Each matching term is added only once to the enhanced query.

    Args:
        query (str): The original query string

    Returns:
        str: The enhanced query with technical terms added
    """

    tech_terms_set = set()
    try:

        possible_paths = [
            "filtered_tech_terms.txt",
            "./filtered_tech_terms.txt",
            "../filtered_tech_terms.txt",
            "C:/Users/Andrea/Desktop/retrieval-pipeline/filtered_tech_terms.txt",
        ]

        file_found = False
        for path in possible_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    print(f"Successfully found and opened: {path}")
                    for line in f:
                        term = line.strip()
                        if term:
                            tech_terms_set.add(term)
                    file_found = True
                    break
            except FileNotFoundError:
                continue

        if not file_found:
            raise FileNotFoundError(
                "Could not find filtered_tech_terms.txt in any expected location"
            )

    except FileNotFoundError:
        print("Warning: filtered_tech_terms.txt not found. Query will not be enhanced.")
        return query

    print(f"Loaded {len(tech_terms_set)} technical terms from filtered_tech_terms.txt")

    words = query.lower().split()

    matching_terms = set()
    for term in tech_terms_set:
        if term.lower() in query.lower():
            matching_terms.add(term)

    enhanced_query = query
    if matching_terms:
        enhanced_query += " " + " ".join(matching_terms)

    return enhanced_query


def parsf():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_file", default="../data/corpora/pubmed.md")
    p.add_argument("--question_file", default="../data/questions_df.csv")
    p.add_argument(
        "--chunker",
        default="fixed_token_chunker.FixedTokenChunker",
        help="module.ClassName",
    )
    p.add_argument(
        "--level",
        choices=["token", "char"],
        help="Evaluation level: token or char",
        default="token",
    )
    p.add_argument("--chunk_size", type=int, default=400)
    p.add_argument("--chunk_overlap", type=int, default=125)
    p.add_argument("--embed_model", default="multi-qa-mpnet-base-dot-v1")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    return args
