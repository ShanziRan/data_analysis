from pathlib import Path

import pandas as pd

try:
    from tools.answer_parsing import is_mcq_answer_key
except ModuleNotFoundError:
    # Allows running this file directly (python tools/split_confidence_teleform.py).
    from answer_parsing import is_mcq_answer_key


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for encoding in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Unable to read {csv_path} with a supported encoding')


def normalize_confidence(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.dropna().empty:
        return numeric
    if numeric.dropna().max() > 1:
        numeric = numeric / 100.0
    return numeric


def main() -> None:
    input_path = Path('data/main_data.csv')
    reference_no_human_path = Path('data/all_no_human.csv')
    if not input_path.exists():
        raise FileNotFoundError('Expected data/main_data.csv to exist.')

    output_dir = Path('data/hitl')
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_csv_with_fallback(input_path)
    data = data.copy()
    data['confidence'] = normalize_confidence(data['confidence'])

    threshold = 0.85
    if 'scan_id' not in data.columns:
        raise KeyError("Column 'scan_id' is required in main_data.csv")

    # Teleform-level rule:
    # - human if any row in a scan_id has confidence <= threshold (or missing confidence)
    # - human if any MCQ row in a scan_id has multiple captured answers
    # - no_human only if all rows in the scan_id are > threshold
    low_or_missing = data['confidence'].le(threshold) | data['confidence'].isna()
    answer_key_is_mcq = data['ANSWER Key'].apply(is_mcq_answer_key)
    captured_norm = data['Captured'].astype(str).str.strip()
    blank_capture = captured_norm.str.casefold() == '--blank--'
    mcq_multiple_answers = answer_key_is_mcq & (captured_norm.str.len() > 1) & (~blank_capture)

    human_scan_ids = set(data.loc[low_or_missing | mcq_multiple_answers, 'scan_id'])

    human_df = data[data['scan_id'].isin(human_scan_ids)].copy()
    no_human_df = data[~data['scan_id'].isin(human_scan_ids)].copy()

    human_path = output_dir / 'human.csv'
    no_human_path = output_dir / 'no_human.csv'

    human_df.to_csv(human_path, index=False)
    no_human_df.to_csv(no_human_path, index=False)

    print(f'Loaded {len(data)} rows from {input_path}')
    print(f'Saved {len(human_df)} rows to {human_path}')
    print(f'Saved {len(no_human_df)} rows to {no_human_path}')

    print(f'Human teleforms (scan_id count): {human_df["scan_id"].nunique()}')
    print(f'No-human teleforms (scan_id count): {no_human_df["scan_id"].nunique()}')

    if reference_no_human_path.exists():
        reference = read_csv_with_fallback(reference_no_human_path)
        if 'scan_id' not in reference.columns:
            raise KeyError("Column 'scan_id' is required in all_no_human.csv")

        current_scan_ids = set(no_human_df['scan_id'])
        reference_scan_ids = set(reference['scan_id'])

        only_in_current = current_scan_ids - reference_scan_ids
        only_in_reference = reference_scan_ids - current_scan_ids

        print('\nComparison with data/all_no_human.csv')
        print(f'Reference rows: {len(reference)}')
        print(f'Reference teleforms (scan_id count): {reference["scan_id"].nunique()}')
        print(f'Generated no_human rows: {len(no_human_df)}')
        print(f'Generated no_human teleforms (scan_id count): {len(current_scan_ids)}')
        print(f'scan_id only in generated no_human: {len(only_in_current)}')
        print(f'scan_id only in reference all_no_human: {len(only_in_reference)}')
    else:
        print('Skipped reference check: data/all_no_human.csv not found.')


if __name__ == '__main__':
    main()
