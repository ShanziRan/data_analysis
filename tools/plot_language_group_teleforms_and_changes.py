from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def count_character_edits(change_cell):
    text = str(change_cell).strip()
    if not text or text.casefold() == 'nan':
        return 0
    return sum(bool(change.strip()) for change in text.split(';'))


def resolve_language_group_column(df):
    for column_name in df.columns:
        normalised = ''.join(char for char in str(column_name).casefold() if char.isalnum())
        if normalised == 'languagegroup':
            return column_name
    raise ValueError('Missing required column: Language group')


def format_donut_labels(total, top_n=6):
    slice_index = 0

    def format_label(percentage):
        nonlocal slice_index
        label = '' if slice_index >= top_n else f'{percentage:.1f}%\n({round(percentage / 100 * total)})'
        slice_index += 1
        return label

    return format_label


def top_group_labels(counts, top_n=6):
    top_groups = set(counts.head(top_n).index)
    return [group if group in top_groups else '' for group in counts.index]


def plot_language_group_teleforms_and_changes(csv_paths, output_path):
    frames = [pd.read_csv(csv_path, encoding='utf-8', engine='python') for csv_path in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    required_columns = {'scan_id', 'character_changes'}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing_columns))}")

    language_group_column = resolve_language_group_column(df)
    df['language_group'] = df[language_group_column].fillna('UNKNOWN').astype(str).str.strip()
    df.loc[df['language_group'].eq(''), 'language_group'] = 'UNKNOWN'
    df['character_edit_count'] = df['character_changes'].map(count_character_edits)

    teleforms_by_group = df.groupby('language_group')['scan_id'].nunique().sort_values(ascending=False)
    changes_by_group = df.groupby('language_group')['character_edit_count'].sum().sort_values(ascending=False)
    language_groups = sorted(set(teleforms_by_group.index).union(changes_by_group.index))
    colour_map = plt.get_cmap('tab20', len(language_groups))
    group_colours = {group: colour_map(index) for index, group in enumerate(language_groups)}

    total_teleforms = int(teleforms_by_group.sum())
    total_changes = int(changes_by_group.sum())

    print('Teleforms by language group:')
    print(teleforms_by_group.to_string())
    print(f'Total teleforms across language groups: {total_teleforms}')
    print('Character changes by language group:')
    print(changes_by_group.to_string())
    print(f'Total character changes: {total_changes}')

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].pie(
        teleforms_by_group.values,
        labels=top_group_labels(teleforms_by_group),
        autopct=format_donut_labels(total_teleforms),
        colors=[group_colours[group] for group in teleforms_by_group.index],
        startangle=90,
        wedgeprops={'width': 0.45, 'edgecolor': 'white'},
        textprops={'fontsize': 20},
    )
    axes[0].text(0, 0, f'{total_teleforms}\nteleforms', ha='center', va='center', fontsize=20)
    axes[0].set_title('Teleforms by Language Group')

    axes[1].pie(
        changes_by_group.values,
        labels=top_group_labels(changes_by_group),
        autopct=format_donut_labels(total_changes),
        colors=[group_colours[group] for group in changes_by_group.index],
        startangle=90,
        wedgeprops={'width': 0.45, 'edgecolor': 'white'},
        textprops={'fontsize': 20},
    )
    axes[1].text(0, 0, f'{total_changes}\nchanges', ha='center', va='center', fontsize=20)
    axes[1].set_title('Character Changes by Language Group')

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f'Saved language-group donut charts to {output_path}')


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
    OUTPUT_PATH = 'output/char_error/new_data/language_group_teleforms_and_changes.png'

    plot_language_group_teleforms_and_changes(CSV_PATHS, OUTPUT_PATH)
