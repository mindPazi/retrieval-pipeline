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


def compute_precision_recall(questions_df, results, chunked_corpus, full_text):
    """
    Calculate and print token-level Precision and Recall
    """
    # Lists to accumulate precision and recall for each question
    all_precisions = []
    all_recalls = []

    # Iterate over all questions (one row for each list of reference spans)
    for i in range(len(questions_df["references"])):

        # Extract the reference spans (list of (start, end)) for question i
        reference_spans = questions_df["references"].iloc[i]
        # Merge any overlapping intervals in the references
        reference_ranges = union_ranges(reference_spans)

        # Get the indices of the predicted chunks for the same question
        pred_idxs = results[
            i
        ]  # we can do this because the length of results and question_df is the same

        # Build the list of intervals (start, end) for the predicted chunks
        predicted_ranges = []
        for idx in pred_idxs:
            if idx < len(chunked_corpus):
                chunk_text = chunked_corpus[idx]
                match = rigorous_document_search(full_text, chunk_text)
                if match:
                    _, start, end = match
                    predicted_ranges.append((start, end))

        # Merge any overlapping intervals among the predicted chunks
        predicted_ranges = union_ranges(predicted_ranges)

        # Calculate the intersection between reference_ranges and predicted_ranges
        intersection_ranges = []
        for r_ref in reference_ranges:
            for r_pred in predicted_ranges:
                intersect = intersect_two_ranges(r_ref, r_pred)
                if intersect:
                    intersection_ranges.append(intersect)
        intersection_ranges = union_ranges(intersection_ranges)

        # Count the "tokens" (actually characters) in each set
        t_r = sum_of_ranges(predicted_ranges)  # total characters retrieved
        t_e = sum_of_ranges(reference_ranges)  # total characters expected
        t_e_and_t_r = sum_of_ranges(intersection_ranges)  # total correct characters

        # Calculate precision and recall (handle zero-division)
        precision = t_e_and_t_r / t_r if t_r > 0 else 0
        recall = t_e_and_t_r / t_e if t_e > 0 else 0

        # Add the metrics to the lists
        all_precisions.append(precision)
        all_recalls.append(recall)

    # Calculate averages across all questions
    avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0

    # Return the aggregated metrics
    return avg_precision, avg_recall


# # Print debug info for question i
#         print(f"--- Question {i} ---")
#         print(f"Question: {questions_df['question'].iloc[i]}")
#         print(f"Best chunk match: {chunked_corpus[pred_idxs[0]]}")
#         print(
#             f"Expected answer: {full_text[reference_spans[0][0]:reference_spans[0][1]]}"
#         )
#         print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
#         print("-" * 60)
