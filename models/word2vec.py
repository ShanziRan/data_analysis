import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools.answer_parsing import expand_answer_variations, is_mcq_answer_key
from tools.question_type import filter_by_language_group, filter_by_part_number, filter_by_pos

def load_analysis_data(csv_path='data/hitl/human.csv'):
    for encoding in ('utf-8', 'cp1252', 'latin-1'):
        try:
            data = pd.read_csv(csv_path, encoding=encoding)
            print(f'Successfully read {csv_path} with encoding: {encoding}')
            return data
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Unable to read {csv_path} with a supported encoding')


def _tokenize(text):
    text = str(text).strip().casefold()
    if not text:
        return []

    tokens = re.findall(r'(?u)\b\w+\b', text)
    if tokens:
        return tokens

    # Fallback for symbol-heavy inputs where word tokenization yields no terms.
    return [char for char in text if not char.isspace()]


def _suffix_token(value):
    return str(value).strip().lower().replace('/', '-').replace(' ', '_')


_EMBEDDING_MODEL_CACHE = {}


def load_pretrained_glove_model(model_name='glove-wiki-gigaword-100'):
    if model_name in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[model_name]

    try:
        import gensim.downloader as api
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            'Word2Vec analysis requires gensim. Install it with: pip install gensim'
        ) from error

    model = api.load(model_name)
    _EMBEDDING_MODEL_CACHE[model_name] = model
    return model


def _sentence_vector(text, model):
    vectors_obj = model.wv if hasattr(model, 'wv') else model
    tokens = _tokenize(text)
    vectors = [vectors_obj[token] for token in tokens if token in vectors_obj]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def _cosine_similarity(left_vec, right_vec):
    left_norm = np.linalg.norm(left_vec)
    right_norm = np.linalg.norm(right_vec)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


def best_word2vec_cosine_match(captured_value, accepted_answers, model):
    captured_text = str(captured_value).strip()
    if not captured_text or not accepted_answers:
        return None, None

    captured_vec = _sentence_vector(captured_text, model)
    if captured_vec is None:
        return None, None

    best_variation = None
    best_score = None

    for answer in accepted_answers:
        candidate_text = str(answer).strip()
        if not candidate_text:
            continue

        candidate_vec = _sentence_vector(candidate_text, model)
        if candidate_vec is None:
            continue

        score = _cosine_similarity(captured_vec, candidate_vec)
        if score is None:
            continue

        if best_score is None or score > best_score:
            best_variation = candidate_text
            best_score = score

    return best_variation, best_score


def run_word2vec_analysis(
    row_start,
    row_end,
    skipped_rows=None,
    glove_model_name='glove-wiki-gigaword-100',
    part_numbers=None,
    pos_values=None,
    language_groups=None,
    include_unmapped_part_number=False,
    part_number_csv_path='data/part_number.csv',
):
    global data
    data = load_analysis_data()
    run_started_at = time.strftime('%Y-%m-%d %H:%M:%S')
    output_log = []
    output_dir = Path('output') / 'vector'
    file_suffix = ''
    if part_numbers:
        file_suffix = '_part_' + '_'.join(_suffix_token(value) for value in part_numbers)
    if pos_values:
        file_suffix = f"{file_suffix}_pos_" + '_'.join(_suffix_token(value) for value in pos_values)
    if language_groups:
        file_suffix = f"{file_suffix}_lang_" + '_'.join(_suffix_token(value) for value in language_groups)

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}{file_suffix}_glove"
    run_dir = output_dir / 'new_data' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / 'word2vec_analysis_info.json'
    row_csv_path = run_dir / 'word2vec_analysis_rows.csv'

    skip_set = set(skipped_rows or [])
    analysis_columns = [
        'number_variations',
        'exact_match',
        'exact_matched_variation',
        'best_variation',
        'max_cosine_similarity',
        'multiple_choice',
    ]
    analysis_df = pd.DataFrame(index=data.index, columns=analysis_columns)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting answer key processing")
    answer_key_start = time.perf_counter()

    selected_rows = data.iloc[row_start:row_end].copy()
    selected_rows = selected_rows.drop(index=skip_set, errors='ignore')

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

    if language_groups:
        pre_filter_count = len(selected_rows)
        selected_rows = filter_by_language_group(selected_rows, language_groups)
        message = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Language-group filter enabled ({language_groups}): "
            f"kept {len(selected_rows)} of {pre_filter_count} rows"
        )
        print(message)
        output_log.append({'type': 'info', 'message': message})

    selected_answers = selected_rows['ANSWER Key'].dropna()
    selected_answers = selected_answers.drop(index=skip_set, errors='ignore')
    selected_answer_count = len(selected_answers)

    expanded_variations = {}

    for idx, answer in selected_answers.items():
        accepted_answers = expand_answer_variations(answer)
        expanded_variations[idx] = accepted_answers
        analysis_df.at[idx, 'number_variations'] = len(accepted_answers)
        print(f"Row {idx}: {len(accepted_answers)} variations")

    answer_key_elapsed = time.perf_counter() - answer_key_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished answer key processing in {answer_key_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading pretrained GloVe model: {glove_model_name}")
    model_load_start = time.perf_counter()
    model = load_pretrained_glove_model(glove_model_name)
    model_load_elapsed = time.perf_counter() - model_load_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded pretrained GloVe model in {model_load_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting GloVe cosine similarity calculation")
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
            best_variation, max_cosine_similarity = best_word2vec_cosine_match(captured_value, accepted_answers, model)

        is_multiple_choice = is_mcq_answer_key(data.loc[idx, 'ANSWER Key'])

        analysis_df.at[idx, 'exact_match'] = exact_match
        analysis_df.at[idx, 'exact_matched_variation'] = exact_matched_variation
        analysis_df.at[idx, 'best_variation'] = best_variation
        analysis_df.at[idx, 'max_cosine_similarity'] = max_cosine_similarity
        analysis_df.at[idx, 'multiple_choice'] = is_multiple_choice

        uid_value = data.loc[idx, 'UID']
        confidence_value = data.loc[idx, 'confidence']
        number_variations = len(accepted_answers)
        message = (
            f"Row {idx+2}: UID = {uid_value}, confidence = {confidence_value}, "
            f"exact_match = {exact_match}, exact_matched_variation = {exact_matched_variation}, "
            f"captured = {captured_value}, best variation = {best_variation}, "
            f"max cosine similarity = {max_cosine_similarity}, "
            f"multiple_choice = {is_multiple_choice}, number_variations = {number_variations}"
        )
        # print(message)

    similarity_elapsed = time.perf_counter() - similarity_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished GloVe cosine similarity calculation in {similarity_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    run_ended_at = time.strftime('%Y-%m-%d %H:%M:%S')
    summary_payload = {
        'run_started_at': run_started_at,
        'run_ended_at': run_ended_at,
        'settings': {
            'row_start': row_start,
            'row_end': row_end,
            'glove_model_name': glove_model_name,
            'part_numbers': part_numbers,
            'pos_values': pos_values,
            'language_groups': language_groups,
            'include_unmapped_part_number': include_unmapped_part_number,
            'part_number_csv_path': part_number_csv_path,
            'skipped_rows_count': len(skip_set),
            'input_csv_path': csv_path,
        },
        'counts': {
            'selected_rows_after_filters': len(selected_rows),
            'selected_answers_non_null': selected_answer_count,
            'processed_rows': len(expanded_variations),
        },
        'timings_seconds': {
            'answer_key_processing': round(answer_key_elapsed, 4),
            'glove_model_loading': round(model_load_elapsed, 4),
            'similarity_calculation': round(similarity_elapsed, 4),
            'total': round(answer_key_elapsed + model_load_elapsed + similarity_elapsed, 4),
        },
        'messages': output_log,
    }

    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    print(f'Saved output log to {output_path}')

    export_df = pd.concat([data, analysis_df], axis=1)
    processed_index = sorted(expanded_variations.keys())
    export_df = export_df.loc[processed_index].copy()
    export_df.to_csv(row_csv_path, index=False)
    print(f'Saved analysis CSV to {row_csv_path}')


if __name__ == '__main__':
    # Set row range here (0-indexed) and specify any rows to skip (0-indexed)
    # ROW_START = 39027
    # ROW_END = 189120
    # Row range excluding D822/03 for no_human
    # ROW_START = 35977
    # ROW_END = 334034
    # Row range excluding D822/03 for human
    ROW_START = 0
    ROW_END = -1
    SKIP_ROWS = []
    GLOVE_MODEL_NAME = 'glove-wiki-gigaword-100'
    PART_NUMBERS = None  # Optional list, e.g. ['1'] or ['1', '2']
    POS_SELECTION = ['D432/01', 'D441/01', 'D441/03']  # Optional list, e.g. ['822/03', 'D441/01']
    LANGUAGE_GROUPS = None  # Optional list, e.g. ['ENGLISH', 'HINDI']
    INCLUDE_UNMAPPED_PART_NUMBER = False
    PART_NUMBER_CSV_PATH = 'data/part_number.csv'

    run_word2vec_analysis(
        row_start=ROW_START,
        row_end=ROW_END,
        skipped_rows=SKIP_ROWS,
        glove_model_name=GLOVE_MODEL_NAME,
        part_numbers=PART_NUMBERS,
        pos_values=POS_SELECTION,
        language_groups=LANGUAGE_GROUPS,
        include_unmapped_part_number=INCLUDE_UNMAPPED_PART_NUMBER,
        part_number_csv_path=PART_NUMBER_CSV_PATH,
    )
