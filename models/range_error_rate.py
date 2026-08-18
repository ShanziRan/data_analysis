import json
from pathlib import Path

import pandas as pd

# For a given range of confidence file, calculate the OCR error rate (Published != Captured)

data_dir = Path('data')
range_files = sorted(data_dir.glob('confidence_range_*.csv'))
output_log_path = data_dir / 'range_error_rate_log.json'

if not range_files:
	raise FileNotFoundError(
		"No confidence range files found. Run confidence_thres.py first to generate files in data/."
	)

total_rows = 0
total_errors = 0
range_logs = []

print('OCR error rate by confidence range:')
for file_path in range_files:
	df = pd.read_csv(file_path)

	# Keep rows where both answers exist for a fair comparison.
	valid_df = df.dropna(subset=['Published', 'Captured']).copy()
	valid_rows = len(valid_df)

	if valid_rows == 0:
		print(f"- {file_path.name}: no valid rows to compare")
		range_logs.append(
			{
				'range_file': file_path.name,
				'valid_rows': 0,
				'errors': 0,
				'error_rate': None,
				'first_error_pair': None,
			}
		)
		continue

	published = valid_df['Published'].astype(str).str.strip()
	captured = valid_df['Captured'].astype(str).str.strip()
	mismatch_mask = published != captured

	errors = int(mismatch_mask.sum())
	error_rate = (errors / valid_rows) * 100

	total_rows += valid_rows
	total_errors += errors

	if errors > 0:
		first_error_row = valid_df.loc[mismatch_mask].iloc[0]
		first_error_pair = {
			'Published': str(first_error_row['Published']).strip(),
			'Captured': str(first_error_row['Captured']).strip(),
		}
		print(
			f"- {file_path.name}: {error_rate:.2f}% "
			f"({errors}/{valid_rows} mismatches), "
			f"first error pair = ({first_error_pair['Published']}, {first_error_pair['Captured']})"
		)
	else:
		first_error_pair = None
		print(
			f"- {file_path.name}: {error_rate:.2f}% "
			f"({errors}/{valid_rows} mismatches), first error pair = none"
		)

	range_logs.append(
		{
			'range_file': file_path.name,
			'valid_rows': valid_rows,
			'errors': errors,
			'error_rate': round(error_rate, 6),
			# 'first_error_pair': first_error_pair,
		}
	)

if total_rows == 0:
	print('\nOverall: no valid rows found across files.')
	overall_log = {
		'overall_error_rate': None,
		'total_errors': 0,
		'total_rows': 0,
	}
else:
	overall_error_rate = (total_errors / total_rows) * 100
	print(
		f"\nOverall error rate: {overall_error_rate:.2f}% "
		f"({total_errors}/{total_rows} mismatches)"
	)
	overall_log = {
		'overall_error_rate': round(overall_error_rate, 6),
		'total_errors': total_errors,
		'total_rows': total_rows,
	}

log_payload = {
	'ranges': range_logs,
	'overall': overall_log,
}

with output_log_path.open('w', encoding='utf-8') as handle:
	json.dump(log_payload, handle, indent=2)

print(f"Saved range error-rate log to {output_log_path}")