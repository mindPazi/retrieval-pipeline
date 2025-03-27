import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def find_query_despite_whitespace(document, query):

    normalized_query = re.sub(r"\s+", " ", query).strip()

    pattern = r"\s*".join(re.escape(word) for word in normalized_query.split())

    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)

    if match:
        return document[match.start() : match.end()], match.start(), match.end()
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


from transformers import AutoTokenizer


def compute_token_level(
    questions_df,
    results,
    chunked_corpus,
    full_text,
    tokenizer_name="distilbert-base-uncased",
):
    """
    Calculate token-level Precision e Recall invece di char-level.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoding = tokenizer(full_text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    def char_to_token_span(start_char, end_char):
        start_token = next(
            (i for i, (s, e) in enumerate(offsets) if e > start_char), len(offsets)
        )
        end_token = next(
            (i for i, (s, e) in enumerate(offsets) if s >= end_char), len(offsets)
        )
        return (start_token, end_token)

    all_precisions, all_recalls = [], []

    for i, ref_list in enumerate(questions_df["references"]):

        ref_ranges = union_ranges(ref_list)
        ref_token_ranges = [char_to_token_span(s, e) for s, e in ref_ranges]

        pred_idxs = results[i]
        pred_token_ranges = []
        for idx in pred_idxs:
            match = rigorous_document_search(full_text, chunked_corpus[idx])
            if match:
                _, s, e = match
                pred_token_ranges.append(char_to_token_span(s, e))
        pred_token_ranges = union_ranges(pred_token_ranges)

        intersect = union_ranges(
            [
                intersect_two_ranges(r_ref, r_pred)
                for r_ref in ref_token_ranges
                for r_pred in pred_token_ranges
                if intersect_two_ranges(r_ref, r_pred) is not None
            ]
        )

        t_r = sum_of_ranges(pred_token_ranges)
        t_e = sum_of_ranges(ref_token_ranges)
        t_e_and_t_r = sum_of_ranges(intersect)

        precision = t_e_and_t_r / t_r if t_r else 0
        recall = t_e_and_t_r / t_e if t_e else 0

        all_precisions.append(precision)
        all_recalls.append(recall)

    return sum(all_precisions) / len(all_precisions), sum(all_recalls) / len(
        all_recalls
    )
