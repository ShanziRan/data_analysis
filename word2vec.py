import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from tools.answer_parsing import expand_answer_variations, is_mcq_answer_key

csv_path = 'data/confidence_gt_85.csv'
for encoding in ('utf-8', 'cp1252', 'latin-1'):
    try:
        data = pd.read_csv(csv_path, encoding=encoding)
        print(f'Successfully read {csv_path} with encoding: {encoding}')
        break
    except UnicodeDecodeError:
        continue
else:
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


def _sentence_vector(text, model):
    tokens = _tokenize(text)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
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


def run_word2vec_analysis(row_start, row_end, skipped_rows=None):
    output_log = []
    output_dir = Path('output') / 'vector'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'word2vec_analysis_{row_start}_{row_end}_no_human_new.json'
    row_csv_path = output_dir / f'word2vec_analysis_{row_start}_{row_end}_no_human_new.csv'

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

    selected_answers = data['ANSWER Key'].iloc[row_start:row_end].dropna()
    selected_answers = selected_answers.drop(index=skip_set, errors='ignore')

    expanded_variations = {}
    corpus = []

    for idx, answer in selected_answers.items():
        accepted_answers = expand_answer_variations(answer)
        expanded_variations[idx] = accepted_answers
        analysis_df.at[idx, 'number_variations'] = len(accepted_answers)
        print(f"Row {idx}: {len(accepted_answers)} variations")

        for candidate in accepted_answers:
            tokens = _tokenize(candidate)
            if tokens:
                corpus.append(tokens)

        captured_value = str(data.loc[idx, 'Captured']).strip()
        if captured_value and captured_value.casefold() != '--blank--':
            tokens = _tokenize(captured_value)
            if tokens:
                corpus.append(tokens)

    answer_key_elapsed = time.perf_counter() - answer_key_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished answer key processing in {answer_key_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    if not corpus:
        raise ValueError('No valid tokenized text found to train Word2Vec.')

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Word2Vec training")
    training_start = time.perf_counter()

    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        epochs=20,
        seed=42,
    )

    training_elapsed = time.perf_counter() - training_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished Word2Vec training in {training_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Word2Vec cosine similarity calculation")
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
        })

    similarity_elapsed = time.perf_counter() - similarity_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished Word2Vec cosine similarity calculation in {similarity_elapsed:.2f} seconds"
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
ROW_START = 72678
ROW_END = 462182
SKIP_ROWS = []

run_word2vec_analysis(row_start=ROW_START, row_end=ROW_END, skipped_rows=SKIP_ROWS)
