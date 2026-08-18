from pathlib import Path

import pandas as pd


def main() -> None:
    csv_path = Path('data/all_no_human.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f'File not found: {csv_path}')

    df = pd.read_csv(csv_path, encoding='cp1252')
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')

    count = int((df['confidence'] < 0.85).sum())
    print(f'Rows with confidence < 0.85: {count}')


if __name__ == '__main__':
    main()
