from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def count_character_edits(change_cell):
    text = str(change_cell).strip()
    if not text or text.casefold() == 'nan':
        return 0
    return sum(bool(change.strip()) for change in text.split(';'))


def split_character_changes(change_cell):
    text = str(change_cell).strip()
    if not text or text.casefold() == 'nan':
        return []
    return [change.strip() for change in text.split(';') if change.strip()]


def count_changes(csv_paths):
    frames = [pd.read_csv(csv_path, encoding='utf-8', engine='python') for csv_path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    if 'character_changes' not in df.columns:
        raise ValueError("Missing required column: character_changes")
    return int(df['character_changes'].map(count_character_edits).sum())


def character_change_table(csv_paths, top_n=20):
    frames = [pd.read_csv(csv_path, encoding='utf-8', engine='python') for csv_path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    if 'character_changes' not in df.columns:
        raise ValueError("Missing required column: character_changes")

    changes = [
        change
        for change_cell in df['character_changes']
        for change in split_character_changes(change_cell)
    ]
    total_changes = len(changes)
    counts = pd.Series(changes, dtype='object').value_counts()
    if top_n is not None and top_n > 0:
        counts = counts.head(top_n)

    rows = []
    for change, count in counts.items():
        before, after = (part.strip() for part in change.split('->', maxsplit=1))
        rows.append({
            'Character before': before,
            'Character after': after,
            '% of errors': f'{count / total_changes * 100:.1f}%',
        })
    return pd.DataFrame(rows), total_changes


def plot_character_change_tables(box_csv_paths, sentence_csv_paths, output_path, top_n=20):
    box_table, box_total = character_change_table(box_csv_paths, top_n=top_n)
    sentence_table, sentence_total = character_change_table(sentence_csv_paths, top_n=top_n)
    max_rows = max(len(box_table), len(sentence_table), 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, max_rows * 0.45 + 2)))
    for ax, table_data, title, total in (
        (axes[0], box_table, 'Box-Letter Responses', box_total),
        (axes[1], sentence_table, 'Sentence Responses', sentence_total),
    ):
        ax.axis('off')
        table = ax.table(
            cellText=table_data.values,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.4)
        ax.set_title(f'{title} ({total} character changes)', fontsize=16)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved response-type character-change tables to {output_path}')


def plot_character_changes_by_response_type(
    box_csv_paths,
    sentence_csv_paths,
    output_path,
    table_output_path=None,
    table_top_n=20,
):
    box_changes = count_changes(box_csv_paths)
    sentence_changes = count_changes(sentence_csv_paths)
    total_changes = box_changes + sentence_changes

    print(f'Box-response character changes: {box_changes}')
    print(f'Sentence-response character changes: {sentence_changes}')
    print(f'Total character changes: {total_changes}')

    labels = ['Box responses', 'Sentence responses']
    values = [box_changes, sentence_changes]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        values,
        labels=labels,
        autopct=lambda percentage: f'{percentage:.1f}%\n({round(percentage / 100 * total_changes)})',
        startangle=180,
        wedgeprops={'width': 0.45, 'edgecolor': 'white'},
        textprops={'fontsize': 20},
    )
    ax.text(0, 0, f'{total_changes}\nchanges', ha='center', va='center', fontsize=20)
    ax.set_title('Character Changes by Response Type')
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f'Saved response-type character-change chart to {output_path}')

    if table_output_path is not None:
        plot_character_change_tables(
            box_csv_paths,
            sentence_csv_paths,
            table_output_path,
            top_n=table_top_n,
        )


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
    OUTPUT_PATH = 'output/char_error/new_data/response_type.png'
    TABLE_OUTPUT_PATH = 'output/char_error/new_data/response_type_character_change_table.png'
    TABLE_TOP_N = 10  # Set to None or <=0 to include all change patterns.

    plot_character_changes_by_response_type(
        BOX_CSV_PATHS,
        SENTENCE_CSV_PATHS,
        OUTPUT_PATH,
        table_output_path=TABLE_OUTPUT_PATH,
        table_top_n=TABLE_TOP_N,
    )
