import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from tools.char_error import extract_character_changes
from tools.question_type import filter_by_part_number, filter_by_pos, filter_by_language_group


def read_csv_with_fallback(csv_path):
    for encoding in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Unable to read {csv_path} with a supported encoding')


def _suffix_token(value):
    return str(value).strip().lower().replace('/', '-').replace(' ', '_')


def _append_suffix_to_filename(filename, suffix):
    if not suffix:
        return filename
    path = Path(filename)
    return f'{path.stem}{suffix}{path.suffix}'


def run_character_change_analysis(
    input_csv_path='data/hitl/human.csv',
    output_root='output/char_error/new_data',
    output_csv_name='captured_published_char_changes.csv',
    summary_json_name='captured_published_char_change_summary.json',
    pos_values=None,
    part_numbers=None,
    language_groups=None,
    include_unmapped_part_number=False,
    part_number_csv_path='data/part_number.csv',
):
    run_started_at = time.strftime('%Y-%m-%d %H:%M:%S')
    started = time.perf_counter()

    input_csv_path = Path(input_csv_path)
    file_suffix = ''
    if part_numbers:
        part_suffix = '_part_' + '_'.join(_suffix_token(value) for value in part_numbers)
        file_suffix = f'{file_suffix}{part_suffix}'
    if pos_values:
        pos_suffix = '_pos_' + '_'.join(_suffix_token(value) for value in pos_values)
        file_suffix = f'{file_suffix}{pos_suffix}'
    if language_groups:
        lang_suffix = '_lang_' + '_'.join(_suffix_token(value) for value in language_groups)
        file_suffix = f'{file_suffix}{lang_suffix}'

    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}{file_suffix}"
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_output_csv_name = _append_suffix_to_filename(output_csv_name, file_suffix)
    resolved_summary_json_name = _append_suffix_to_filename(summary_json_name, file_suffix)

    output_csv_path = run_dir / resolved_output_csv_name
    summary_json_path = run_dir / resolved_summary_json_name

    if not input_csv_path.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_csv_path}')

    df = read_csv_with_fallback(input_csv_path)
    original_row_count = len(df)

    if pos_values:
        df = filter_by_pos(df, pos_values)

    if part_numbers:
        df = filter_by_part_number(
            df,
            part_numbers=part_numbers,
            include_unmapped=include_unmapped_part_number,
            part_number_csv_path=part_number_csv_path,
            output_column='inferred_part_number',
        )

    if language_groups:
        df = filter_by_language_group(
            df,
            language_groups
        )

    required_columns = ['Captured', 'Published']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required column(s): {', '.join(missing_columns)}")

    change_counter = Counter()
    character_changes_column = []
    rows_with_character_changes = 0

    for _, row in df.iterrows():
        captured_value = str(row['Captured']).strip()
        published_value = str(row['Published']).strip()

        captured_norm = captured_value.casefold()
        published_norm = published_value.casefold()

        if (
            captured_norm == '--blank--'
            or published_norm == '--blank--'
            or not captured_value
            or not published_value
            or captured_norm == 'nan'
            or published_norm == 'nan'
        ):
            row_changes = []
        else:
            row_changes = extract_character_changes(captured_value, published_value)

        if row_changes:
            rows_with_character_changes += 1
            change_counter.update(row_changes)

        character_changes_column.append('; '.join(row_changes))

    output_df = df.copy()
    output_df['character_changes'] = character_changes_column
    output_df.to_csv(output_csv_path, index=False)

    elapsed = time.perf_counter() - started
    run_ended_at = time.strftime('%Y-%m-%d %H:%M:%S')

    summary_payload = {
        'run_started_at': run_started_at,
        'run_ended_at': run_ended_at,
        'settings': {
            'input_csv_path': str(input_csv_path),
            'run_dir': str(run_dir),
            'output_root': str(output_root),

            'pos_values': pos_values,
            'part_numbers': part_numbers,
            'language_groups': language_groups,
            'include_unmapped_part_number': include_unmapped_part_number,
            'part_number_csv_path': part_number_csv_path,
        },
        'counts': {
            'original_rows': original_row_count,
            'rows_after_filters': len(df),
            'total_rows': len(output_df),
            'rows_with_character_changes': rows_with_character_changes,
        },
        'character_change_counts': dict(sorted(change_counter.items(), key=lambda item: (-item[1], item[0]))),
        'timings_seconds': {
            'total': round(elapsed, 4),
        },
    }

    with summary_json_path.open('w', encoding='utf-8') as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    print(f'Saved row-level character changes to {output_csv_path}')
    print(f'Saved character-change summary to {summary_json_path}')


if __name__ == '__main__':
    # Direct-run configuration
    # INPUT_CSV_PATH = 'data/hitl/human.csv'
    INPUT_CSV_PATH = 'output/distance/new_data/run_20260807_145858_872_non_mcq_only_part_4_pos_d432-01/distance_analysis_rows.csv'
    OUTPUT_ROOT = 'output/char_error/new_data/distance_processed'
    OUTPUT_CSV_NAME = 'char_changes.csv'
    SUMMARY_JSON_NAME = 'char_change_summary.json'
    POS_VALUES = ['D432/01']  # Example: ['822/03', 'D441/01'] or None
    PART_NUMBERS = ['4']  # Example: ['1', '2'] or None
    # POS_VALUES = ['D432/01']  # Example: ['822/03', 'D441/01'] or None
    # PART_NUMBERS = ['2', '3']  # Example: ['1', '2'] or None
    INCLUDE_UNMAPPED_PART_NUMBER = False
    PART_NUMBER_CSV_PATH = 'data/part_number.csv'

    run_character_change_analysis(
        input_csv_path=INPUT_CSV_PATH,
        output_root=OUTPUT_ROOT,
        output_csv_name=OUTPUT_CSV_NAME,
        summary_json_name=SUMMARY_JSON_NAME,
        pos_values=POS_VALUES,
        part_numbers=PART_NUMBERS,
        include_unmapped_part_number=INCLUDE_UNMAPPED_PART_NUMBER,
        part_number_csv_path=PART_NUMBER_CSV_PATH,
    )
