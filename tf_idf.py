import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tools.answer_parsing import expand_answer_variations, is_mcq_answer_key
from tools.question_type import (
    add_inferred_question_type_column,
    filter_by_part_number,
    filter_by_pos,
)

csv_path = 'data/hitl/human.csv'
for encoding in ('utf-8', 'cp1252', 'latin-1'):
    try:
        data = pd.read_csv(csv_path, encoding=encoding)
        print(f'Successfully read {csv_path} with encoding: {encoding}')
        break
    except UnicodeDecodeError:
        continue
else:
    raise ValueError(f'Unable to read {csv_path} with a supported encoding')


def best_tfidf_cosine_match(captured_value, accepted_answers):
    captured_text = str(captured_value).strip()
    if not captured_text or not accepted_answers:
        return None, None

    candidate_texts = [str(answer).strip() for answer in accepted_answers if str(answer).strip()]
    if not candidate_texts:
        return None, None

    documents = [captured_text] + candidate_texts

    # Fit a per-row TF-IDF space so captured text is compared only with this row's answer variants.
    # Include single-character tokens (e.g., MCQ answers like A/B/C).
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Fallback for symbol-heavy inputs where word tokenization yields an empty vocabulary.
        vectorizer = TfidfVectorizer(lowercase=True, analyzer='char', ngram_range=(1, 3))
        matrix = vectorizer.fit_transform(documents)

    captured_vector = matrix[0:1]
    candidate_vectors = matrix[1:]
    scores = cosine_similarity(captured_vector, candidate_vectors).flatten()

    best_idx = int(scores.argmax())
    return candidate_texts[best_idx], float(scores[best_idx])


def _suffix_token(value):
    return str(value).strip().lower().replace('/', '-').replace(' ', '_')


def run_tfidf_analysis(
    row_start,
    row_end,
    skipped_rows=None,
    part_numbers=None,
    pos_values=None,
    include_unmapped_part_number=False,
    part_number_csv_path='data/part_number.csv',
):
    output_log = []
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    file_suffix = ''
    if part_numbers:
        file_suffix = '_part_' + '_'.join(_suffix_token(value) for value in part_numbers)
    if pos_values:
        file_suffix = f"{file_suffix}_pos_" + '_'.join(_suffix_token(value) for value in pos_values)
    output_path = output_dir / f'vector/new_data/tfidf_{row_start}_{row_end}_human{file_suffix}.json'
    row_csv_path = output_dir / f'vector/new_data/tfidf_{row_start}_{row_end}_human{file_suffix}.csv'

    skip_set = set(skipped_rows or [])
    analysis_columns = [
        'number_variations',
        'exact_match',
        'exact_matched_variation',
        'best_variation',
        'max_cosine_similarity',
        'multiple_choice',
        'inferred_question_type',
    ]
    analysis_df = pd.DataFrame(index=data.index, columns=analysis_columns)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting answer key processing")
    answer_key_start = time.perf_counter()

    selected_rows = data.iloc[row_start:row_end].copy()
    selected_rows = selected_rows.drop(index=skip_set, errors='ignore')
    selected_rows = add_inferred_question_type_column(
        selected_rows,
        part_number_csv_path=part_number_csv_path,
        output_column='inferred_question_type',
    )

    if pos_values:
        pre_filter_count = len(selected_rows)
        selected_rows = filter_by_pos(selected_rows, pos_values)
        message = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] PoS filter enabled ({pos_values}): "
            f"kept {len(selected_rows)} of {pre_filter_count} rows"
        )
        print(message)
        output_log.append({'type': 'info', 'message': message})

    if part_numbers:
        pre_filter_count = len(selected_rows)
        selected_rows = filter_by_part_number(
            selected_rows,
            part_numbers=part_numbers,
            include_unmapped=include_unmapped_part_number,
            part_number_csv_path=part_number_csv_path,
            output_column='inferred_part_number',
        )
        message = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Part-number filter enabled ({part_numbers}): "
            f"kept {len(selected_rows)} of {pre_filter_count} rows"
        )
        print(message)
        output_log.append({'type': 'info', 'message': message})

    selected_answers = selected_rows['ANSWER Key'].dropna()

    expanded_variations = {}
    for idx, answer in selected_answers.items():
        accepted_answers = expand_answer_variations(answer)
        expanded_variations[idx] = accepted_answers
        analysis_df.at[idx, 'number_variations'] = len(accepted_answers)
        analysis_df.at[idx, 'inferred_question_type'] = selected_rows.at[idx, 'inferred_question_type']
        print(f"Row {idx}: {len(accepted_answers)} variations")

    answer_key_elapsed = time.perf_counter() - answer_key_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished answer key processing in {answer_key_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting TF-IDF cosine similarity calculation")
    similarity_start = time.perf_counter()

    for idx, accepted_answers in expanded_variations.items():
        captured_value = str(data.loc[idx, 'Captured']).strip()
        captured_norm = captured_value.casefold()
        exact_matched_variation = None
        for candidate in accepted_answers:
            candidate_text = str(candidate).strip()
            if captured_norm and candidate_text.casefold() == captured_norm:
                exact_matched_variation = candidate_text
                break
        exact_match = exact_matched_variation is not None

        if captured_norm == '--blank--':
            best_variation = None
            max_cosine_similarity = 0.0
        elif exact_match:
            best_variation = exact_matched_variation
            max_cosine_similarity = 1.0
        else:
            best_variation, max_cosine_similarity = best_tfidf_cosine_match(captured_value, accepted_answers)

        is_multiple_choice = is_mcq_answer_key(data.loc[idx, 'ANSWER Key'])

        analysis_df.at[idx, 'exact_match'] = exact_match
        analysis_df.at[idx, 'exact_matched_variation'] = exact_matched_variation
        analysis_df.at[idx, 'best_variation'] = best_variation
        analysis_df.at[idx, 'max_cosine_similarity'] = max_cosine_similarity
        analysis_df.at[idx, 'multiple_choice'] = is_multiple_choice

        uid_value = data.loc[idx, 'UID']
        confidence_value = data.loc[idx, 'confidence']
        number_variations = len(accepted_answers)
        inferred_question_type = analysis_df.at[idx, 'inferred_question_type']
        message = (
            f"Row {idx+2}: UID = {uid_value}, confidence = {confidence_value}, "
            f"exact_match = {exact_match}, exact_matched_variation = {exact_matched_variation}, "
            f"captured = {captured_value}, best variation = {best_variation}, "
            f"max cosine similarity = {max_cosine_similarity}, "
            f"multiple_choice = {is_multiple_choice}, inferred_question_type = {inferred_question_type}, "
            f"number_variations = {number_variations}"
        )
        print(message)
        output_log.append({
            'row': idx + 2,
            'UID': uid_value,
            'confidence': confidence_value,
            'captured': captured_value,
            'number_variations': number_variations,
            'exact_match': exact_match,
            'exact_matched_variation': exact_matched_variation,
            'best_variation': best_variation,
            'max_cosine_similarity': max_cosine_similarity,
            'multiple_choice': is_multiple_choice,
            'inferred_question_type': inferred_question_type,
        })

    similarity_elapsed = time.perf_counter() - similarity_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished TF-IDF cosine similarity calculation in {similarity_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(output_log, handle, indent=2, ensure_ascii=False)

    print(f'Saved output log to {output_path}')

    export_df = pd.concat([data, analysis_df], axis=1)
    processed_index = sorted(expanded_variations.keys())
    export_df = export_df.loc[processed_index].copy()
    export_df.to_csv(row_csv_path, index=False)
    print(f'Saved analysis CSV to {row_csv_path}')


# Set row range here (0-indexed) and specify any rows to skip (0-indexed)
# ROW_START = 39027
# ROW_END = 189120
# ROW_START = 39027
# ROW_END = 145968
ROW_START = 35977
ROW_END = 334034
SKIP_ROWS = []
PART_NUMBERS = None  # Optional list, e.g. ['1'] or ['1', '2']
POS_SELECTION = None  # Optional list, e.g. ['822/03'] or ['D822/03']
INCLUDE_UNMAPPED_PART_NUMBER = False

run_tfidf_analysis(
    row_start=ROW_START,
    row_end=ROW_END,
    skipped_rows=SKIP_ROWS,
    part_numbers=PART_NUMBERS,
    pos_values=POS_SELECTION,
    include_unmapped_part_number=INCLUDE_UNMAPPED_PART_NUMBER,
)
