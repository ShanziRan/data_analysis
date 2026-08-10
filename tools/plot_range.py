from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    csv_path = Path('output/range/summary_overview.csv')
    output_path = Path('output/range/error_type_trends.png')

    df = pd.read_csv(csv_path)
    required_columns = {'confidence_range', 'category', 'category_name', 'percentage'}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(f'Missing required columns: {sorted(missing_columns)}')

    df = df.copy()
    df['confidence_range'] = df['confidence_range'].astype(str)

    ranges = list(dict.fromkeys(df['confidence_range']))
    pivot = df.pivot_table(index='confidence_range', columns='category', values='percentage', aggfunc='first').reindex(ranges)

    category_names = df.groupby('category')['category_name'].first().to_dict()
    x_positions = list(range(len(ranges)))

    plt.figure(figsize=(10, 6))
    for category in sorted(pivot.columns):
        label = category_names.get(category, f'Category {category}')
        plt.plot(x_positions, pivot[category].tolist(), marker='o', linewidth=1.8, label=label)

    plt.xticks(x_positions, ranges, rotation=45, ha='right')
    plt.xlabel('Confidence range')
    plt.ylabel('Percentage of errors')
    plt.title('Error category percentages by confidence range')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()

