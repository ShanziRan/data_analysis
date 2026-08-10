import json
import time
from pathlib import Path

import pandas as pd

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


def levenshtein_distance(source, target):
    # Normalise the input strings by stripping whitespace and converting to lowercase
    source = str(source).strip().casefold()
    target = str(target).strip().casefold()

    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current_row = [i]
        for j, target_char in enumerate(target, start=1):
            insertion_cost = current_row[j - 1] + 1
            deletion_cost = previous_row[j] + 1
            substitution_cost = previous_row[j - 1] + (0 if source_char == target_char else 1)
            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
        previous_row = current_row

    return previous_row[-1]


def damerau_levenshtein_distance(source, target):
    # Normalise the input strings by stripping whitespace and converting to lowercase
    source = str(source).strip().casefold()
    target = str(target).strip().casefold()

    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    len_source = len(source)
    len_target = len(target)
    max_dist = len_source + len_target

    matrix = [[0] * (len_target + 2) for _ in range(len_source + 2)]
    matrix[0][0] = max_dist

    for i in range(len_source + 1):
        matrix[i + 1][0] = max_dist
        matrix[i + 1][1] = i

    for j in range(len_target + 1):
        matrix[0][j + 1] = max_dist
        matrix[1][j + 1] = j

    last_row = {}

    for i in range(1, len_source + 1):
        last_match_col = 0
        for j in range(1, len_target + 1):
            i1 = last_row.get(target[j - 1], 0)
            j1 = last_match_col
            cost = 0 if source[i - 1] == target[j - 1] else 1

            if source[i - 1] == target[j - 1]:
                last_match_col = j
                matrix[i + 1][j + 1] = min(
                    matrix[i][j] + cost,
                    matrix[i + 1][j] + 1,
                    matrix[i][j + 1] + 1,
                    matrix[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
                )
            else:
                matrix[i + 1][j + 1] = min(
                    matrix[i][j] + cost,
                    matrix[i + 1][j] + 1,
                    matrix[i][j + 1] + 1,
                    matrix[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
                )

        last_row[source[i - 1]] = i

    return matrix[len_source + 1][len_target + 1]


def _suffix_token(value):
    return str(value).strip().lower().replace('/', '-').replace(' ', '_')


def run_distance_analysis(
    row_start,
    row_end,
    skipped_rows=None,
    non_mcq_only=False,
    check_human=False,
    part_numbers=None,
    pos_values=None,
    include_unmapped_part_number=False,
    part_number_csv_path='data/part_number.csv',
):
    run_started_at = time.strftime('%Y-%m-%d %H:%M:%S')
    output_log = []
    output_dir = Path('output')
    file_suffix = '_non_mcq_only' if non_mcq_only else ''
    if part_numbers:
        part_suffix = '_part_' + '_'.join(_suffix_token(value) for value in part_numbers)
        file_suffix = f'{file_suffix}{part_suffix}'
    if pos_values:
        pos_suffix = '_pos_' + '_'.join(_suffix_token(value) for value in pos_values)
        file_suffix = f'{file_suffix}{pos_suffix}'
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}{file_suffix}"
    run_dir = output_dir / 'distance/new_data' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / 'distance_analysis_info.json'
    row_csv_path = run_dir / 'distance_analysis_rows.csv'

    skip_set = set(skipped_rows or [])
    analysis_columns = [
        'number_variations',
        'exact_match',
        'exact_matched_variation',
        'min_distance',
        'published_min_distance',
        'published_distance_gt_captured',
        'best_variation',
        'normalised_similarity',
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
    selected_answer_count = len(selected_answers)

    expanded_variations = {}
    skipped_mcq_count = 0
    for idx, answer in selected_answers.items():
        accepted_answers = expand_answer_variations(answer)
        is_multiple_choice = is_mcq_answer_key(answer)

        if non_mcq_only and is_multiple_choice:
            skipped_mcq_count += 1
            continue

        expanded_variations[idx] = accepted_answers
        analysis_df.at[idx, 'number_variations'] = len(accepted_answers)
        analysis_df.at[idx, 'inferred_question_type'] = selected_rows.at[idx, 'inferred_question_type']
        message = f"Row {idx}: {len(accepted_answers)} variations"
        print(message)

    if non_mcq_only:
        message = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Non-MCQ mode enabled: "
            f"skipped {skipped_mcq_count} MCQ rows from the selected range"
        )
        print(message)
        output_log.append({'type': 'info', 'message': message})

    answer_key_elapsed = time.perf_counter() - answer_key_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished answer key processing in {answer_key_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting minimum distance calculation")
    distance_start = time.perf_counter()

    for idx, accepted_answers in expanded_variations.items():
        captured_value = str(data.loc[idx, 'Captured']).strip()
        captured_norm = captured_value.casefold()
        published_value = str(data.loc[idx, 'Published']).strip()
        published_norm = published_value.casefold()
        exact_matched_variation = None

        for candidate in accepted_answers:
            candidate_text = str(candidate).strip()
            if captured_norm and candidate_text.casefold() == captured_norm:
                exact_matched_variation = candidate_text
                break

        exact_match = exact_matched_variation is not None

        if not accepted_answers or not captured_value:
            min_distance = None
            best_variation = None
        elif exact_match:
            min_distance = 0
            best_variation = exact_matched_variation
        else:
            min_distance = None
            best_variation = None
            for candidate in accepted_answers:
                candidate_text = str(candidate).strip()
                distance = levenshtein_distance(captured_value, candidate_text)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    best_variation = candidate_text

        is_multiple_choice = is_mcq_answer_key(data.loc[idx, 'ANSWER Key'])

        if captured_value.casefold() == '--blank--':
            if is_multiple_choice:
                min_distance = 1
            elif accepted_answers:
                min_distance = min(len(str(answer).strip()) for answer in accepted_answers)
            else:
                min_distance = None
            similarity = 0.0
        elif min_distance is None or not captured_value:
            similarity = None
        elif exact_match:
            similarity = 1.0
        else:
            max_len = max(len(captured_value), len(best_variation or ""))
            similarity = 1.0 - (min_distance / max_len) if max_len else 1.0

        published_min_distance = None
        published_distance_gt_captured = False
        if check_human and captured_norm != published_norm:
            if accepted_answers and published_value:
                published_min_distance = min(
                    levenshtein_distance(published_value, str(candidate).strip())
                    for candidate in accepted_answers
                )
            if published_min_distance is not None and min_distance is not None:
                published_distance_gt_captured = published_min_distance > min_distance

        # similarity = 1.0 - (min_distance / len(best_variation)) if best_variation else 1.0

        analysis_df.at[idx, 'min_distance'] = min_distance
        analysis_df.at[idx, 'published_min_distance'] = published_min_distance
        analysis_df.at[idx, 'published_distance_gt_captured'] = published_distance_gt_captured
        analysis_df.at[idx, 'best_variation'] = best_variation
        analysis_df.at[idx, 'exact_match'] = exact_match
        analysis_df.at[idx, 'exact_matched_variation'] = exact_matched_variation
        analysis_df.at[idx, 'normalised_similarity'] = similarity
        analysis_df.at[idx, 'multiple_choice'] = is_multiple_choice

        uid_value = data.loc[idx, 'UID']
        confidence_value = data.loc[idx, 'confidence']
        number_variations = len(accepted_answers)
        inferred_question_type = analysis_df.at[idx, 'inferred_question_type']
        message = (
            f"Row {idx+2}: UID = {uid_value}, confidence = {confidence_value}, "
            f"exact_match = {exact_match}, exact_matched_variation = {exact_matched_variation}, "
            f"captured = {captured_value}, min distance = {min_distance}, "
            f"published = {published_value}, published min distance = {published_min_distance}, "
            f"published_distance_gt_captured = {published_distance_gt_captured}, "
            f"best variation = {best_variation}, similarity = {similarity}, "
            f"multiple_choice = {is_multiple_choice}, inferred_question_type = {inferred_question_type}, "
            f"number_variations = {number_variations}"
        )
        # print(message)
    distance_elapsed = time.perf_counter() - distance_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished minimum distance calculation in {distance_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    run_ended_at = time.strftime('%Y-%m-%d %H:%M:%S')
    summary_payload = {
        'run_started_at': run_started_at,
        'run_ended_at': run_ended_at,
        'settings': {
            'row_start': row_start,
            'row_end': row_end,
            'non_mcq_only': non_mcq_only,
            'check_human': check_human,
            'part_numbers': part_numbers,
            'pos_values': pos_values,
            'include_unmapped_part_number': include_unmapped_part_number,
            'part_number_csv_path': part_number_csv_path,
            'skipped_rows_count': len(skip_set),
        },
        'counts': {
            'selected_rows_after_filters': len(selected_rows),
            'selected_answers_non_null': selected_answer_count,
            'processed_rows': len(expanded_variations),
        },
        'timings_seconds': {
            'answer_key_processing': round(answer_key_elapsed, 4),
            'distance_calculation': round(distance_elapsed, 4),
            'total': round(answer_key_elapsed + distance_elapsed, 4),
        },
        'messages': output_log,
    }

    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    print(f"Saved output log to {output_path}")

    export_df = pd.concat([data, analysis_df], axis=1)
    processed_index = sorted(expanded_variations.keys())
    export_df = export_df.loc[processed_index].copy()
    export_df.to_csv(row_csv_path, index=False)
    print(f"Saved analysis CSV to {row_csv_path}")


if __name__ == '__main__':
    # Direct-run configuration
    ROW_START = 0
    ROW_END = -1  # Use -1 to include all remaining rows
    SKIP_ROWS = []  # Example: [12, 85]
    NON_MCQ_ONLY = True
    CHECK_HUMAN = True
    PART_NUMBERS = ['4']  # Example: ['1', '2']
    POS_VALUES = ['D432/01']  # Example: ['D822/03', 'D441/01']
    INCLUDE_UNMAPPED_PART_NUMBER = False
    PART_NUMBER_CSV_PATH = 'data/part_number.csv'

    run_distance_analysis(
        row_start=ROW_START,
        row_end=ROW_END,
        skipped_rows=SKIP_ROWS,
        non_mcq_only=NON_MCQ_ONLY,
        check_human=CHECK_HUMAN,
        part_numbers=PART_NUMBERS,
        pos_values=POS_VALUES,
        include_unmapped_part_number=INCLUDE_UNMAPPED_PART_NUMBER,
        part_number_csv_path=PART_NUMBER_CSV_PATH,
    )
