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

# Collect all OCR error rows (Published != Captured) and save them to a separate CSV file for further analysis.
error_dir = Path('error')

# Keep rows where both fields exist, then compare normalized text values.
valid_data = data.dropna(subset=['Published', 'Captured']).copy()
published = valid_data['Published'].astype(str).str.strip()
captured = valid_data['Captured'].astype(str).str.strip()

error_mask = published != captured
error_rows = valid_data.loc[error_mask].copy()

# output_path = error_dir / 'all_ocr_errors.csv'
# error_rows.to_csv(output_path, index=False)

# print(f'Total valid rows compared: {len(valid_data)}')
# print(f'Total OCR error rows: {len(error_rows)}')
# print(f'Saved OCR errors to {output_path}')

# Analyse the OCR error rows by type of error
# TODO
error_dir.mkdir(exist_ok=True)

error_rows['_published_norm'] = error_rows['Published'].astype(str).str.strip()
error_rows['_captured_norm'] = error_rows['Captured'].astype(str).str.strip()
published_len = error_rows['_published_norm'].str.len()
captured_len = error_rows['_captured_norm'].str.len()

# 1) Multiple choice with extra captured characters.
multi_choice_multiple_answers = error_rows.loc[(published_len == 1) & (captured_len > 1)].copy()

# 2) Multiple choice wrong recognition.
multi_choice_wrong_recognition = error_rows.loc[(published_len == 1) & (captured_len == 1)].copy()

# 3) Misspelling in non-single-character answers.
misspelling_errors = error_rows.loc[(published_len > 1) & (captured_len > 1)].copy()

multi_choice_multiple_answers.drop(columns=['_published_norm', '_captured_norm'], inplace=True, errors='ignore')
multi_choice_wrong_recognition.drop(columns=['_published_norm', '_captured_norm'], inplace=True, errors='ignore')
misspelling_errors.drop(columns=['_published_norm', '_captured_norm'], inplace=True, errors='ignore')

multi_choice_multiple_answers_path = error_dir / 'multiple_choice_more_than_one_answer.csv'
multi_choice_wrong_recognition_path = error_dir / 'multiple_choice_wrong_recognition.csv'
misspelling_errors_path = error_dir / 'misspelling_errors.csv'

multi_choice_multiple_answers.to_csv(multi_choice_multiple_answers_path, index=False)
multi_choice_wrong_recognition.to_csv(multi_choice_wrong_recognition_path, index=False)
misspelling_errors.to_csv(misspelling_errors_path, index=False)

print(f'Total valid rows compared: {len(valid_data)}')
print(f'Total OCR error rows: {len(error_rows)}')
print(f'Category 1 rows (published single char, captured > 1 char): {len(multi_choice_multiple_answers)}')
print(f'Saved category 1 errors to {multi_choice_multiple_answers_path}')
print(f'Category 2 rows (both single char and different): {len(multi_choice_wrong_recognition)}')
print(f'Saved category 2 errors to {multi_choice_wrong_recognition_path}')
print(f'Category 3 rows (both > 1 char and different): {len(misspelling_errors)}')
print(f'Saved category 3 errors to {misspelling_errors_path}')