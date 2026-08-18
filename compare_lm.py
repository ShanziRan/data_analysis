from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

SCAN_ID_COLUMN = 'scan_id'

LM_COLUMNS_NEAR_CONFIDENCE = ['aws_lm_confidence', 'aws_lm_confidence_final']
LM_COLUMNS_AT_END = ['aws_lm_guess']


def read_csv_with_fallback(csv_path):
    for encoding in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(csv_path, encoding=encoding, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Unable to read {csv_path} with a supported encoding')


def normalise_scan_id(value):
    text = str(value).strip()
    if not text or text.casefold() == 'nan':
        return None

    try:
        return str(Decimal(text).quantize(Decimal('1')))
    except InvalidOperation:
        return text


def compare_lm(distance_csv_path, lm_csv_path='data/lm.csv', output_csv_path=None):
    distance_df = read_csv_with_fallback(distance_csv_path)
    lm_df = read_csv_with_fallback(lm_csv_path)

    missing_distance_cols = [col for col in (SCAN_ID_COLUMN, 'confidence') if col not in distance_df.columns]
    if missing_distance_cols:
        raise ValueError(f"Missing required column(s) in {distance_csv_path}: {', '.join(missing_distance_cols)}")

    missing_lm_cols = [
        col for col in [SCAN_ID_COLUMN] + LM_COLUMNS_NEAR_CONFIDENCE + LM_COLUMNS_AT_END
        if col not in lm_df.columns
    ]
    if missing_lm_cols:
        raise ValueError(f"Missing required column(s) in {lm_csv_path}: {', '.join(missing_lm_cols)}")

    distance_df['_scan_id'] = distance_df[SCAN_ID_COLUMN].map(normalise_scan_id)
    lm_df['_scan_id'] = lm_df[SCAN_ID_COLUMN].map(normalise_scan_id)

    duplicate_lm_scan_ids = lm_df['_scan_id'].duplicated().sum()
    if duplicate_lm_scan_ids:
        print(f'Warning: {duplicate_lm_scan_ids} duplicate scan_id value(s) found in {lm_csv_path}; keeping first occurrence.')
    lm_lookup = lm_df.drop_duplicates(subset='_scan_id', keep='first').set_index('_scan_id')

    lm_columns_to_pull = LM_COLUMNS_NEAR_CONFIDENCE + LM_COLUMNS_AT_END
    matched = distance_df['_scan_id'].notna() & distance_df['_scan_id'].isin(lm_lookup.index)

    for col in lm_columns_to_pull:
        distance_df[col] = distance_df['_scan_id'].map(lm_lookup[col])

    distance_df = distance_df.drop(columns=['_scan_id'])

    # Reorder: place aws_lm_confidence / aws_lm_confidence_final right after 'confidence',
    # and aws_lm_guess at the very end.
    columns = list(distance_df.columns)
    for col in lm_columns_to_pull:
        columns.remove(col)

    confidence_idx = columns.index('confidence')
    columns[confidence_idx + 1:confidence_idx + 1] = LM_COLUMNS_NEAR_CONFIDENCE
    columns.append('aws_lm_guess')
    distance_df = distance_df[columns]

    total_rows = len(distance_df)
    matched_rows = int(matched.sum())
    print(f'Total rows: {total_rows}')
    print(f'Rows matched in {lm_csv_path}: {matched_rows} ({matched_rows / total_rows:.1%})')
    print(f'Rows not found in {lm_csv_path}: {total_rows - matched_rows}')

    if output_csv_path is None:
        distance_path = Path(distance_csv_path)
        output_csv_path = distance_path.with_name(f'{distance_path.stem}_with_lm{distance_path.suffix}')

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distance_df.to_csv(output_path, index=False)
    print(f'Saved enriched CSV to {output_path}')

    return distance_df


if __name__ == '__main__':
    # Direct-run configuration
    DISTANCE_CSV_PATH = (
        'output/distance/new_data/run_20260807_111928_556_non_mcq_only_part_2_pos_d441-03/distance_analysis_rows.csv'
    )
    LM_CSV_PATH = 'data/lm.csv'
    OUTPUT_CSV_PATH = None  # Defaults to <DISTANCE_CSV_PATH>_with_lm.csv

    compare_lm(
        DISTANCE_CSV_PATH,
        lm_csv_path=LM_CSV_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
    )
