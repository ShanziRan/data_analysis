import re
from pathlib import Path

import pandas as pd

from tools.answer_parsing import is_mcq_answer_key

data_dir = Path('data')
range_dir = data_dir / 'ranged'
range_files = sorted(range_dir.glob('confidence_range_*.csv'))
output_dir = Path('output') / 'range'


def read_csv_with_fallback(csv_path):
    for encoding in ('utf-8', 'cp1252', 'latin-1'):
        try:
            data = pd.read_csv(csv_path, encoding=encoding)
            print(f'Successfully read {csv_path} with encoding: {encoding}')
            return data
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Unable to read {csv_path} with a supported encoding')


def analyse_range_errors(data):
    valid_data = data.dropna(subset=['Published', 'Captured']).copy()
    published = valid_data['Published'].astype(str).str.strip()
    captured = valid_data['Captured'].astype(str).str.strip()

    error_mask = published != captured
    error_rows = valid_data.loc[error_mask].copy()

    error_rows['_published_norm'] = error_rows['Published'].astype(str).str.strip()
    error_rows['_captured_norm'] = error_rows['Captured'].astype(str).str.strip()
    error_rows['_answer_key_is_mcq'] = error_rows['ANSWER Key'].apply(is_mcq_answer_key)
    captured_len = error_rows['_captured_norm'].str.len()

    blank_capture_mask = error_rows['_captured_norm'].str.casefold() == '--blank--'
    mcq_blank_capture_mask = error_rows['_answer_key_is_mcq'] & blank_capture_mask
    non_mcq_blank_capture_mask = (~error_rows['_answer_key_is_mcq']) & blank_capture_mask
    multi_choice_multiple_answers_mask = error_rows['_answer_key_is_mcq'] & (captured_len > 1) & (~blank_capture_mask)
    multi_choice_wrong_recognition_mask = error_rows['_answer_key_is_mcq'] & (captured_len == 1)
    misspelling_errors_mask = (~error_rows['_answer_key_is_mcq']) & (captured_len > 1) & (~blank_capture_mask)

    return {
        'valid_rows': len(valid_data),
        'error_rows': len(error_rows),
        'category_1': int(mcq_blank_capture_mask.sum()),
        'category_2': int(non_mcq_blank_capture_mask.sum()),
        'category_3': int(multi_choice_multiple_answers_mask.sum()),
        'category_4': int(multi_choice_wrong_recognition_mask.sum()),
        'category_5': int(misspelling_errors_mask.sum()),
    }


def _to_percent_label(number_text):
    value = float(number_text.replace('_', '.')) * 100
    if value.is_integer():
        return str(int(value))
    return f'{value:.1f}'


def format_range_label(file_stem):
    suffix = file_stem.replace('confidence_range_', '')

    if suffix.startswith('start_'):
        upper = suffix.replace('start_', '', 1)
        return f'0-{_to_percent_label(upper)}'

    if suffix.endswith('_end'):
        lower = suffix.replace('_end', '')
        return f'{_to_percent_label(lower)}-end'

    between_match = re.fullmatch(r'([0-9_]+)_to_([0-9_]+)', suffix)
    if between_match:
        lower = _to_percent_label(between_match.group(1))
        upper = _to_percent_label(between_match.group(2))
        return f'{lower}-{upper}'

    return suffix


if not range_files:
    raise FileNotFoundError(
        'No confidence range files found in data/ranged. Run confidence_thres.py first.'
    )

output_dir.mkdir(parents=True, exist_ok=True)
overview_rows = []

for file_path in range_files:
    range_data = read_csv_with_fallback(file_path)
    analysis = analyse_range_errors(range_data)
    range_label = format_range_label(file_path.stem)
    error_total = analysis['error_rows']

    category_details = [
        (1, 'mcq_blank_captured', analysis['category_1']),
        (2, 'non_mcq_blank_captured', analysis['category_2']),
        (3, 'single_published_multiple_captured', analysis['category_3']),
        (4, 'single_published_single_captured_wrong', analysis['category_4']),
        (5, 'multi_char_published_multi_char_captured', analysis['category_5']),
    ]

    for category_id, category_name, count in category_details:
        percentage = round((count / error_total) * 100, 1) if error_total else 0.0
        overview_rows.append(
            {
                'confidence_range': range_label,
                'category': category_id,
                'category_name': category_name,
                'number': count,
                'percentage': percentage,
            }
        )

    print(f'Range file: {file_path.name}')
    print(f'Total valid rows compared: {analysis["valid_rows"]}')
    print(f'Total OCR error rows: {analysis["error_rows"]}')
    print(f'Category 1 rows (MCQ published, captured --blank--): {analysis["category_1"]}')
    print(f'Category 2 rows (non-MCQ published, captured --blank--): {analysis["category_2"]}')
    print(f'Category 3 rows (published single char, captured multiple chars): {analysis["category_3"]}')
    print(f'Category 4 rows (both single char and different): {analysis["category_4"]}')
    print(f'Category 5 rows (both > 1 char and different, non-blank): {analysis["category_5"]}')

overview_df = pd.DataFrame(overview_rows)
overview_path = output_dir / 'summary_overview.csv'
overview_df.to_csv(overview_path, index=False)
print(f'Saved overview to {overview_path}')