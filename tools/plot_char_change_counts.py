import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_change_counts(json_path):
    with Path(json_path).open('r', encoding='utf-8') as handle:
        payload = json.load(handle)

    counts = payload.get('character_change_counts', {})
    settings = payload.get('settings', {})
    label = Path(json_path).stem

    pos_values = settings.get('pos_values') or []
    part_numbers = settings.get('part_numbers') or []
    if pos_values or part_numbers:
        pos_text = '+'.join(str(value) for value in pos_values) if pos_values else 'all-pos'
        part_text = '+'.join(str(value) for value in part_numbers) if part_numbers else 'all-parts'
        label = f'{part_text} | {pos_text}'

    normalized_counts = {str(key): int(value) for key, value in counts.items()}
    return label, normalized_counts


def _normalise_header(name):
    text = str(name).strip().casefold()
    return ''.join(ch for ch in text if ch.isalnum())


def _resolve_language_group_column(df):
    for column_name in df.columns:
        if _normalise_header(column_name) == 'languagegroup':
            return column_name
    raise ValueError('Missing required column: Language Group')


def _resolve_row_csv_path(json_path, payload):
    settings = payload.get('settings', {}) if isinstance(payload, dict) else {}
    run_dir = settings.get('run_dir')

    if run_dir:
        search_dir = Path(run_dir)
    else:
        search_dir = Path(json_path).parent

    csv_candidates = sorted(search_dir.glob('*.csv'))
    if not csv_candidates:
        raise FileNotFoundError(f'No CSV files found near summary: {json_path}')

    preferred = [
        path for path in csv_candidates
        if 'char_changes' in path.name or 'character_change_rows' in path.name
    ]
    if preferred:
        return preferred[0]

    return csv_candidates[0]


def _count_row_change_operations(change_cell):
    text = str(change_cell).strip()
    if not text or text.casefold() == 'nan':
        return 0
    return len([segment for segment in text.split(';') if segment.strip()])


def _split_row_changes(change_cell):
    text = str(change_cell).strip()
    if not text or text.casefold() == 'nan':
        return []
    return [segment.strip() for segment in text.split(';') if segment.strip()]


def select_changes_for_plot(all_series, top_n=40):
    aggregate = {}
    for _, counts in all_series:
        for change, count in counts.items():
            aggregate[change] = aggregate.get(change, 0) + count

    if top_n is None or top_n <= 0:
        selected = sorted(aggregate.keys())
    else:
        ranked = sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))
        selected = [change for change, _ in ranked[:top_n]]

    return selected


def classify_change(change):
    parts = [segment.strip() for segment in str(change).split('->')]
    if len(parts) != 2:
        return 'other'

    source, target = parts
    if source == '<ins>':
        return 'insertion'
    if target == '<del>':
        return 'deletion'
    return 'substitution'


def aggregate_operation_counts(change_counts):
    aggregated = {
        'substitution': 0,
        'deletion': 0,
        'insertion': 0,
        'other': 0,
    }

    for change, count in change_counts.items():
        change_type = classify_change(change)
        aggregated[change_type] = aggregated.get(change_type, 0) + int(count)

    return aggregated


def _parse_change_pair(change):
    parts = [segment.strip() for segment in str(change).split('->')]
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def aggregate_numeric_alpha_counts(change_counts):
    aggregated = {
        'digit_to_alpha': 0,
        'alpha_to_digit': 0,
    }

    for change, count in change_counts.items():
        source, target = _parse_change_pair(change)
        if source is None or target is None:
            continue

        # This analysis focuses on substitutions between one digit and one alphabetic character.
        if len(source) == 1 and len(target) == 1:
            if source.isdigit() and target.isalpha():
                aggregated['digit_to_alpha'] += int(count)
            elif source.isalpha() and target.isdigit():
                aggregated['alpha_to_digit'] += int(count)

    aggregated['numeric_alpha_total'] = aggregated['digit_to_alpha'] + aggregated['alpha_to_digit']
    return aggregated


def plot_grouped_counts(json_paths, output_path, top_n=40):
    series = [load_change_counts(path) for path in json_paths]
    selected_changes = select_changes_for_plot(series, top_n=top_n)

    if not selected_changes:
        raise ValueError('No character changes found in the supplied JSON files.')

    labels = [label for label, _ in series]
    n_groups = len(selected_changes)
    n_series = len(series)

    x = np.arange(n_groups)
    bar_width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(max(16, n_groups * 0.35), 9))

    for idx, (label, counts) in enumerate(series):
        y_values = [counts.get(change, 0) for change in selected_changes]
        offset = (idx - (n_series - 1) / 2.0) * bar_width
        ax.bar(x + offset, y_values, width=bar_width, label=label)

    ax.set_title('Character Change Counts (Grouped Across Runs)')
    ax.set_xlabel('Character change')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(selected_changes, rotation=75, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f'Saved grouped character-change plot to {output_path}')


def plot_grouped_operation_counts(json_paths, output_path):
    series = [load_change_counts(path) for path in json_paths]

    labels = [label for label, _ in series]
    categories = ['substitution', 'deletion', 'insertion']
    n_groups = len(categories)
    n_series = len(series)

    x = np.arange(n_groups)
    bar_width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, (label, counts) in enumerate(series):
        operation_counts = aggregate_operation_counts(counts)
        y_values = [operation_counts.get(category, 0) for category in categories]
        offset = (idx - (n_series - 1) / 2.0) * bar_width
        ax.bar(x + offset, y_values, width=bar_width, label=label)

    ax.set_title('Character Change Counts by Operation Type')
    ax.set_xlabel('Operation type')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f'Saved grouped operation-type plot to {output_path}')


def plot_grouped_numeric_alpha_counts(json_paths, output_path):
    series = [load_change_counts(path) for path in json_paths]

    categories = ['digit_to_alpha', 'alpha_to_digit', 'numeric_alpha_total']
    category_labels = ['digit -> alpha', 'alpha -> digit', 'total']
    n_groups = len(categories)
    n_series = len(series)

    x = np.arange(n_groups)
    bar_width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(10, 7))

    print('Numeric <-> alphabet substitution counts by run:')
    for idx, (label, counts) in enumerate(series):
        numeric_counts = aggregate_numeric_alpha_counts(counts)
        y_values = [numeric_counts.get(category, 0) for category in categories]
        offset = (idx - (n_series - 1) / 2.0) * bar_width
        ax.bar(x + offset, y_values, width=bar_width, label=label)
        print(
            f"- {label}: digit->alpha={numeric_counts['digit_to_alpha']}, "
            f"alpha->digit={numeric_counts['alpha_to_digit']}, total={numeric_counts['numeric_alpha_total']}"
        )

    ax.set_title('Numeric <-> Alphabet Substitution Counts')
    ax.set_xlabel('Change category')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f'Saved grouped numeric-alpha plot to {output_path}')


def plot_language_group_total_counts(json_paths, output_path, top_n=40):
    language_group_change_counts = {}

    for json_path in json_paths:
        summary_path = Path(json_path)
        with summary_path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)

        row_csv_path = _resolve_row_csv_path(summary_path, payload)
        df = pd.read_csv(row_csv_path, encoding='utf-8', engine='python')

        language_group_col = _resolve_language_group_column(df)
        if 'character_changes' not in df.columns:
            raise ValueError(f"Missing required column 'character_changes' in {row_csv_path}")

        for _, row in df.iterrows():
            group_value = str(row[language_group_col]).strip() or 'UNKNOWN'
            if group_value.casefold() == 'nan':
                group_value = 'UNKNOWN'

            if group_value not in language_group_change_counts:
                language_group_change_counts[group_value] = Counter()

            row_changes = _split_row_changes(row['character_changes'])
            for change in row_changes:
                language_group_change_counts[group_value][change] += 1

    if not language_group_change_counts:
        raise ValueError('No language-group change counts could be computed from the supplied JSON paths.')

    sorted_groups = sorted(
        language_group_change_counts.keys(),
        key=lambda group: (-sum(language_group_change_counts[group].values()), group),
    )

    all_group_total = sum(sum(counts.values()) for counts in language_group_change_counts.values())

    n_subplots = len(sorted_groups)
    n_cols = 2 if n_subplots > 1 else 1
    n_rows = int(np.ceil(n_subplots / n_cols))

    fig_width = max(16, (top_n or 40) * 0.35)
    fig_height = max(6, n_rows * 4.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).ravel()

    cmap = plt.get_cmap('tab20', n_subplots)

    for idx, group in enumerate(sorted_groups):
        ax = axes[idx]
        counts = language_group_change_counts[group]
        group_total = sum(counts.values())
        ranked_changes = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if top_n is not None and top_n > 0:
            ranked_changes = ranked_changes[:top_n]
        selected_changes = [change for change, _ in ranked_changes]
        x = np.arange(len(selected_changes))

        if group_total == 0:
            y_values = [0.0] * len(selected_changes)
        else:
            values = [counts.get(change, 0) for change in selected_changes]
            y_values = [value / group_total * 100.0 for value in values]

        ax.bar(x, y_values, width=0.8, color=cmap(idx))
        ax.set_title(f'{group} ({group_total})')
        ax.set_ylabel('% of group total')
        ax.set_ylim(0, 100.0)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(selected_changes, rotation=75, ha='right')
        ax.set_xlabel('Character change')

    for idx in range(n_subplots, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Top Character Change Types by Language Group', y=1.02)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print('Language-group totals (character changes):')
    for group in sorted_groups:
        counts = language_group_change_counts[group]
        group_total = sum(counts.values())
        print(f'- {group}: {group_total}')
    print(f'- All groups: {all_group_total}')
    print(f'Saved language-group percentage plot to {output_path}')


def main():
    # Direct-run configuration
    # JSON_PATHS = [
    #     'output/char_error/new_data/run_20260807_113813_297_part_2_pos_d441-03/char_change_summary_part_2_pos_d441-03.json',
    #     'output/char_error/new_data/run_20260807_114901_942_part_2_3_pos_d441-01/char_change_summary_part_2_3_pos_d441-01.json',
    #     'output/char_error/new_data/run_20260807_115000_542_part_2_3_pos_d432-01/char_change_summary_part_2_3_pos_d432-01.json',
    # ]
    JSON_PATHS = [
        'output/char_error/new_data/run_20260807_115036_095_part_4_pos_d432-01/char_change_summary_part_4_pos_d432-01.json',
        'output/char_error/new_data/run_20260807_115111_485_part_4_pos_d441-01/char_change_summary_part_4_pos_d441-01.json',
        'output/char_error/new_data/run_20260807_115142_396_part_2_pos_d822-03/char_change_summary_part_2_pos_d822-03.json',
    ]
    # PLOT_MODE = 'per_change'  # 'per_change', 'operation_type', 'numeric_alpha', or 'language_group_total'
    # PLOT_MODE = 'operation_type'
    # PLOT_MODE = 'numeric_alpha'
    PLOT_MODE = 'language_group_total'
    OUTPUT_PATH = f'output/char_error/new_data/char_change_counts_{PLOT_MODE}.png'
    TOP_N = 40  # Set to None or <=0 to plot all changes.

    if PLOT_MODE == 'per_change':
        plot_grouped_counts(
            json_paths=JSON_PATHS,
            output_path=OUTPUT_PATH,
            top_n=TOP_N,
        )
    elif PLOT_MODE == 'operation_type':
        plot_grouped_operation_counts(
            json_paths=JSON_PATHS,
            output_path=OUTPUT_PATH,
        )
    elif PLOT_MODE == 'numeric_alpha':
        plot_grouped_numeric_alpha_counts(
            json_paths=JSON_PATHS,
            output_path=OUTPUT_PATH,
        )
    elif PLOT_MODE == 'language_group_total':
        plot_language_group_total_counts(
            json_paths=JSON_PATHS,
            output_path=OUTPUT_PATH,
            top_n=TOP_N,
        )
    else:
        raise ValueError(
            "PLOT_MODE must be 'per_change', 'operation_type', 'numeric_alpha', or 'language_group_total'"
        )


if __name__ == '__main__':
    main()
