from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def count_character_edits(change_cell):
    text = str(change_cell).strip()
    if not text or text.casefold() == 'nan':
        return 0
    return sum(bool(change.strip()) for change in text.split(';'))


def count_teleforms_with_fewer_than_edits(csv_paths, max_edits=3, output_path=None):
    frames = [pd.read_csv(csv_path, encoding='utf-8', engine='python') for csv_path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    required_columns = {'scan_id', 'character_changes'}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing_columns))}")

    df['character_edit_count'] = df['character_changes'].map(count_character_edits)
    edits_by_scan_id = df.groupby('scan_id')['character_edit_count'].sum()
    selected = edits_by_scan_id[edits_by_scan_id < max_edits]
    total_teleforms = len(edits_by_scan_id)
    low_edit_teleforms = len(selected)
    high_edit_teleforms = total_teleforms - low_edit_teleforms

    print(f'Total teleforms: {total_teleforms}')
    print(f'Teleforms with fewer than {max_edits} character-level edits: {low_edit_teleforms}')
    print(f'Teleforms with {max_edits} or more character-level edits: {high_edit_teleforms}')

    if output_path is not None:
        labels = [f'Fewer than {max_edits} edits', f'{max_edits} or more edits']
        values = [low_edit_teleforms, high_edit_teleforms]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            values,
            labels=labels,
            autopct=lambda percentage: f'{percentage:.1f}%\n({round(percentage / 100 * total_teleforms)})',
            startangle=90,
            wedgeprops={'width': 0.45, 'edgecolor': 'white'},
            textprops={'fontsize': 20},
        )
        ax.text(0, 0, f'{total_teleforms}\nteleforms', ha='center', va='center', fontsize=20)
        ax.set_title(f'Teleforms by Character-Level Edit Count (< {max_edits})')
        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f'Saved teleform edit-count chart to {output_path}')

    return selected


if __name__ == '__main__':
    # Direct-run configuration
    CSV_PATHS = [
        'output/char_error/new_data/run_20260807_113813_297_part_2_pos_d441-03/char_changes_part_2_pos_d441-03.csv',
        'output/char_error/new_data/run_20260807_114901_942_part_2_3_pos_d441-01/char_changes_part_2_3_pos_d441-01.csv',
        'output/char_error/new_data/run_20260807_115000_542_part_2_3_pos_d432-01/char_changes_part_2_3_pos_d432-01.csv',
        'output/char_error/new_data/run_20260807_115036_095_part_4_pos_d432-01/char_changes_part_4_pos_d432-01.csv',
        'output/char_error/new_data/run_20260807_115111_485_part_4_pos_d441-01/char_changes_part_4_pos_d441-01.csv',
        'output/char_error/new_data/run_20260807_115142_396_part_2_pos_d822-03/char_changes_part_2_pos_d822-03.csv',
    ]
    MAX_EDITS = 3
    OUTPUT_PATH = 'output/char_error/new_data/teleform_count_by_character_edit.png'

    count_teleforms_with_fewer_than_edits(
        CSV_PATHS,
        max_edits=MAX_EDITS,
        output_path=OUTPUT_PATH,
    )
