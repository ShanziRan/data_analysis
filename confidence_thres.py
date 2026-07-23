import pandas as pd
from pathlib import Path

csv_path = 'main_data.csv'

for encoding in ('utf-8', 'cp1252', 'latin-1'):
    try:
        data = pd.read_csv(csv_path, encoding=encoding)
        print(f'Successfully read {csv_path} with encoding: {encoding}')
        break
    except UnicodeDecodeError:
        continue
else:
    raise ValueError(f'Unable to read {csv_path} with a supported encoding')

# Test print to check the data shape
print(f'Data shape: {data.shape}')

# Threshold for adjustable confidence level, and store the subset of data to separate CSV files based on the confidence threshold
thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]

output_dir = Path('data')
output_dir.mkdir(exist_ok=True)

# Ensure confidence is numeric before range filtering.
data['confidence'] = pd.to_numeric(data['confidence'], errors='coerce')


def format_range_label(start, end):
    if start is None:
        return f"start_{end:.2f}".replace('.', '_')
    if end is None:
        return f"{start:.2f}_end".replace('.', '_')
    return f"{start:.2f}_to_{end:.2f}".replace('.', '_')


# Build ranges: <first, between each pair, and >=last threshold.
ranges = [(None, thresholds[0])]
ranges.extend((thresholds[i], thresholds[i + 1]) for i in range(len(thresholds) - 1))
ranges.append((thresholds[-1], None))

for start, end in ranges:
    if start is None:
        subset = data[data['confidence'] < end]
    elif end is None:
        subset = data[data['confidence'] >= start]
    else:
        subset = data[(data['confidence'] >= start) & (data['confidence'] < end)]

    range_label = format_range_label(start, end)
    output_path = output_dir / f"confidence_range_{range_label}.csv"
    subset.to_csv(output_path, index=False)
    print(f"Saved {len(subset)} rows to {output_path}")
