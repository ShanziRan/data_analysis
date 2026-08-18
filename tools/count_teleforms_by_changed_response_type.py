import pandas as pd


def has_character_changes(change_cell):
    text = str(change_cell).strip()
    return bool(text) and text.casefold() != 'nan'


def load_changed_scan_ids(csv_paths):
    frames = [pd.read_csv(csv_path, encoding='utf-8', engine='python') for csv_path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    required_columns = {'scan_id', 'character_changes'}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing_columns))}")

    all_scan_ids = set(df['scan_id'].dropna())
    changed_scan_ids = set(df.loc[df['character_changes'].map(has_character_changes), 'scan_id'].dropna())
    return all_scan_ids, changed_scan_ids


def count_teleforms_by_changed_response_type(box_csv_paths, sentence_csv_paths):
    box_scan_ids, box_changed_scan_ids = load_changed_scan_ids(box_csv_paths)
    sentence_scan_ids, sentence_changed_scan_ids = load_changed_scan_ids(sentence_csv_paths)

    all_scan_ids = box_scan_ids | sentence_scan_ids
    box_only = box_changed_scan_ids - sentence_changed_scan_ids
    sentence_only = sentence_changed_scan_ids - box_changed_scan_ids
    both = box_changed_scan_ids & sentence_changed_scan_ids
    no_changes = all_scan_ids - (box_changed_scan_ids | sentence_changed_scan_ids)

    categories = {
        'Only box-response changes': box_only,
        'Only sentence-response changes': sentence_only,
        'Changes in both response types': both,
        'No character changes': no_changes,
    }

    total_teleforms = len(all_scan_ids)
    print(f'Total teleforms: {total_teleforms}')
    for label, scan_ids in categories.items():
        percentage = len(scan_ids) / total_teleforms * 100 if total_teleforms else 0.0
        print(f'{label}: {len(scan_ids)} ({percentage:.1f}%)')

    return categories


if __name__ == '__main__':
    # Direct-run configuration
    BOX_CSV_PATHS = [
        'output/char_error/new_data/run_20260807_113813_297_part_2_pos_d441-03/char_changes_part_2_pos_d441-03.csv',
        'output/char_error/new_data/run_20260807_114901_942_part_2_3_pos_d441-01/char_changes_part_2_3_pos_d441-01.csv',
        'output/char_error/new_data/run_20260807_115000_542_part_2_3_pos_d432-01/char_changes_part_2_3_pos_d432-01.csv',
    ]
    SENTENCE_CSV_PATHS = [
        'output/char_error/new_data/run_20260807_115036_095_part_4_pos_d432-01/char_changes_part_4_pos_d432-01.csv',
        'output/char_error/new_data/run_20260807_115111_485_part_4_pos_d441-01/char_changes_part_4_pos_d441-01.csv',
        'output/char_error/new_data/run_20260807_115142_396_part_2_pos_d822-03/char_changes_part_2_pos_d822-03.csv',
    ]

    count_teleforms_by_changed_response_type(BOX_CSV_PATHS, SENTENCE_CSV_PATHS)
