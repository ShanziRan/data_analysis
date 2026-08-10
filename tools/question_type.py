import argparse
import re
from pathlib import Path

import pandas as pd


def read_csv_with_fallback(csv_path):
	for encoding in ('utf-8', 'cp1252', 'latin-1'):
		try:
			return pd.read_csv(csv_path, encoding=encoding)
		except UnicodeDecodeError:
			continue
	raise ValueError(f'Unable to read {csv_path} with a supported encoding')


def _normalise_header(name):
	return re.sub(r'[^a-z0-9]+', '', str(name).strip().casefold())


def _normalise_pos(value):
	if pd.isna(value):
		return None
	text = str(value).strip()
	if not text:
		return None
	text = text.upper()
	# Some datasets prefix PoS with letters (e.g. D822/03). Strip the prefix for matching.
	text = re.sub(r'^[A-Z]+(?=\d)', '', text)
	return text


def normalise_pos(value):
	return _normalise_pos(value)


def _normalise_question_number(value):
	if pd.isna(value):
		return None

	text = str(value).strip()
	if not text:
		return None

	# Accept values like "10", "10.0", and "Q10" by extracting the first integer.
	match = re.search(r'\d+', text)
	if not match:
		return None

	return int(match.group())


def normalise_question_type(value):
	if pd.isna(value):
		return None
	text = str(value).strip().upper()
	return text if text else None


def _normalise_language_group(value):
	if pd.isna(value):
		return None
	text = str(value).strip().casefold()
	return text if text else None


def _resolve_language_group_column(df):
	for column_name in df.columns:
		if _normalise_header(column_name) == 'languagegroup':
			return column_name
	raise ValueError('Missing required data column: Language group')


def load_question_type_map(part_number_csv_path='data/part_number.csv'):
	mapping_df = read_csv_with_fallback(part_number_csv_path)

	header_map = {_normalise_header(col): col for col in mapping_df.columns}

	pos_col = header_map.get('pos')
	question_col = header_map.get('questionnumber')
	type_col = header_map.get('type')

	missing = [
		name
		for name, col in (
			('PoS', pos_col),
			('Question number', question_col),
			('Type', type_col),
		)
		if col is None
	]
	if missing:
		raise ValueError(
			f"Missing required column(s) in {part_number_csv_path}: {', '.join(missing)}"
		)

	question_type_map = {}
	for _, row in mapping_df.iterrows():
		pos_value = _normalise_pos(row[pos_col])
		question_number = _normalise_question_number(row[question_col])
		question_type = normalise_question_type(row[type_col])

		if pos_value is None or question_number is None or question_type is None:
			continue

		question_type_map[(pos_value, question_number)] = question_type

	return question_type_map


def load_part_number_map(part_number_csv_path='data/part_number.csv'):
	mapping_df = read_csv_with_fallback(part_number_csv_path)

	header_map = {_normalise_header(col): col for col in mapping_df.columns}

	pos_col = header_map.get('pos')
	question_col = header_map.get('questionnumber')
	part_col = header_map.get('partnumber')

	missing = [
		name
		for name, col in (
			('PoS', pos_col),
			('Question number', question_col),
			('Part Number', part_col),
		)
		if col is None
	]
	if missing:
		raise ValueError(
			f"Missing required column(s) in {part_number_csv_path}: {', '.join(missing)}"
		)

	part_number_map = {}
	for _, row in mapping_df.iterrows():
		pos_value = _normalise_pos(row[pos_col])
		question_number = _normalise_question_number(row[question_col])
		part_number = _normalise_question_number(row[part_col])

		if pos_value is None or question_number is None or part_number is None:
			continue

		part_number_map[(pos_value, question_number)] = part_number

	return part_number_map


def infer_question_type(pos_value, question_number, question_type_map):
	key = (_normalise_pos(pos_value), _normalise_question_number(question_number))
	if key[0] is None or key[1] is None:
		return None
	return question_type_map.get(key)


def infer_part_number(pos_value, question_number, part_number_map):
	key = (_normalise_pos(pos_value), _normalise_question_number(question_number))
	if key[0] is None or key[1] is None:
		return None
	return part_number_map.get(key)


def add_inferred_question_type_column(
	df,
	question_type_map=None,
	part_number_csv_path='data/part_number.csv',
	output_column='inferred_question_type',
):
	if question_type_map is None:
		question_type_map = load_question_type_map(part_number_csv_path)

	if 'PoS' not in df.columns or 'question_number' not in df.columns:
		missing = [col for col in ('PoS', 'question_number') if col not in df.columns]
		raise ValueError(f"Missing required data column(s): {', '.join(missing)}")

	inferred_types = [
		infer_question_type(pos_value, question_number, question_type_map)
		for pos_value, question_number in zip(df['PoS'], df['question_number'])
	]

	output_df = df.copy()
	output_df[output_column] = inferred_types
	return output_df


def add_inferred_part_number_column(
	df,
	part_number_map=None,
	part_number_csv_path='data/part_number.csv',
	output_column='inferred_part_number',
):
	if part_number_map is None:
		part_number_map = load_part_number_map(part_number_csv_path)

	if 'PoS' not in df.columns or 'question_number' not in df.columns:
		missing = [col for col in ('PoS', 'question_number') if col not in df.columns]
		raise ValueError(f"Missing required data column(s): {', '.join(missing)}")

	inferred_parts = [
		infer_part_number(pos_value, question_number, part_number_map)
		for pos_value, question_number in zip(df['PoS'], df['question_number'])
	]

	output_df = df.copy()
	output_df[output_column] = inferred_parts
	return output_df


def filter_by_question_type(
	df,
	question_types,
	include_unmapped=False,
	question_type_map=None,
	part_number_csv_path='data/part_number.csv',
	output_column='inferred_question_type',
):
	if not question_types:
		return add_inferred_question_type_column(
			df,
			question_type_map=question_type_map,
			part_number_csv_path=part_number_csv_path,
			output_column=output_column,
		)

	wanted_types = {normalise_question_type(value) for value in question_types}
	wanted_types.discard(None)

	enriched_df = add_inferred_question_type_column(
		df,
		question_type_map=question_type_map,
		part_number_csv_path=part_number_csv_path,
		output_column=output_column,
	)

	mask = enriched_df[output_column].isin(wanted_types)
	if include_unmapped:
		mask = mask | enriched_df[output_column].isna()

	return enriched_df.loc[mask].copy()


def filter_by_pos(df, pos_values):
	if not pos_values:
		return df.copy()

	if 'PoS' not in df.columns:
		raise ValueError('Missing required data column: PoS')

	wanted_pos = {normalise_pos(value) for value in pos_values}
	wanted_pos.discard(None)

	normalised_series = df['PoS'].map(normalise_pos)
	mask = normalised_series.isin(wanted_pos)
	return df.loc[mask].copy()


def filter_by_language_group(df, language_groups):
	if not language_groups:
		return df.copy()

	column_name = _resolve_language_group_column(df)
	wanted_groups = {_normalise_language_group(value) for value in language_groups}
	wanted_groups.discard(None)

	normalised_series = df[column_name].map(_normalise_language_group)
	mask = normalised_series.isin(wanted_groups)
	return df.loc[mask].copy()


def filter_by_part_number(
	df,
	part_numbers,
	include_unmapped=False,
	part_number_map=None,
	part_number_csv_path='data/part_number.csv',
	output_column='inferred_part_number',
):
	if not part_numbers:
		return add_inferred_part_number_column(
			df,
			part_number_map=part_number_map,
			part_number_csv_path=part_number_csv_path,
			output_column=output_column,
		)

	wanted_parts = {_normalise_question_number(value) for value in part_numbers}
	wanted_parts.discard(None)

	enriched_df = add_inferred_part_number_column(
		df,
		part_number_map=part_number_map,
		part_number_csv_path=part_number_csv_path,
		output_column=output_column,
	)

	mask = enriched_df[output_column].isin(wanted_parts)
	if include_unmapped:
		mask = mask | enriched_df[output_column].isna()

	return enriched_df.loc[mask].copy()


def _build_cli_parser():
	parser = argparse.ArgumentParser(
		description='Infer question type for data rows based on data/part_number.csv mapping.'
	)
	parser.add_argument('--input-csv', default='data/hitl/human.csv', help='Input CSV path')
	parser.add_argument('--part-number-csv', default='data/part_number.csv', help='Part number mapping CSV path')
	parser.add_argument('--start', type=int, default=0, help='Inclusive 0-based row start')
	parser.add_argument('--end', type=int, default=None, help='Exclusive 0-based row end')
	parser.add_argument(
		'--language-groups',
		nargs='*',
		default=None,
		help='Optional language-group filter, e.g. ENGLISH HINDI',
	)
	parser.add_argument('--pos', nargs='*', default=None, help='Optional PoS filter, e.g. 822/03 D441/01')
	parser.add_argument('--part-numbers', nargs='*', default=None, help='Optional part-number filter, e.g. 1 2 3')
	parser.add_argument(
		'--include-unmapped',
		action='store_true',
		help='When --part-numbers is set, include rows with unknown inferred part number',
	)
	parser.add_argument('--output-csv', default=None, help='Optional path to save enriched/filter result')
	return parser


def main():
	args = _build_cli_parser().parse_args()

	source_df = read_csv_with_fallback(args.input_csv)
	sliced_df = source_df.iloc[args.start:args.end].copy()
	sliced_df = add_inferred_question_type_column(
		sliced_df,
		part_number_csv_path=args.part_number_csv,
	)

	if args.language_groups:
		sliced_df = filter_by_language_group(sliced_df, args.language_groups)

	if args.pos:
		sliced_df = filter_by_pos(sliced_df, args.pos)

	if args.part_numbers:
		result_df = filter_by_part_number(
			sliced_df,
			part_numbers=args.part_numbers,
			include_unmapped=args.include_unmapped,
			part_number_csv_path=args.part_number_csv,
		)
	else:
		result_df = add_inferred_part_number_column(
			sliced_df,
			part_number_csv_path=args.part_number_csv,
		)

	print(f'Total rows scanned: {len(sliced_df)}')
	print(f'Rows returned: {len(result_df)}')
	print('Inferred part-number counts:')
	print(result_df['inferred_part_number'].value_counts(dropna=False))
	print('Inferred type counts:')
	print(result_df['inferred_question_type'].value_counts(dropna=False))

	if args.output_csv:
		output_path = Path(args.output_csv)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		result_df.to_csv(output_path, index=False)
		print(f'Saved output CSV to {output_path}')


if __name__ == '__main__':
	main()
