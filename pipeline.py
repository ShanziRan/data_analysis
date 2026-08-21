import argparse
import re
import time
from pathlib import Path

import pandas as pd

from models.edit_distance import levenshtein_distance
from models.word2vec import best_word2vec_cosine_match, load_pretrained_glove_model
from tools.answer_parsing import expand_answer_variations, is_mcq_answer_key


def read_csv_with_fallback_encodings(csv_path):
	for encoding in ('utf-8', 'cp1252', 'latin-1'):
		try:
			data = pd.read_csv(csv_path, encoding=encoding)
			print(f'Successfully read {csv_path} with encoding: {encoding}')
			return data
		except UnicodeDecodeError:
			continue
	raise ValueError(f'Unable to read {csv_path} with a supported encoding')


def _text(value):
	return '' if pd.isna(value) else str(value).strip()


def classify_mcq(captured_value, answer_key):
	captured_norm = captured_value.casefold()
	answer_norm = _text(answer_key).casefold()
	if not captured_value or captured_norm == '--blank--':
		return 'mcq_no_answer'

	answer_tokens = re.findall(r'\b[A-Za-z]\b', captured_value)
	if len(answer_tokens) > 1:
		return 'mcq_multiple_answers'
	if captured_norm == answer_norm:
		return 'mcq_exact_match'
	return 'mcq_incorrect'


def _distance_metrics(captured_value, accepted_answers, answer_key):
	captured_norm = captured_value.casefold()
	exact_matched_variation = next(
		(
			str(answer).strip()
			for answer in accepted_answers
			if captured_norm and str(answer).strip().casefold() == captured_norm
		),
		None,
	)
	exact_match = exact_matched_variation is not None

	if captured_norm == '--blank--':
		min_distance = 1 if is_mcq_answer_key(answer_key) else min(
			(len(str(answer).strip()) for answer in accepted_answers),
			default=None,
		)
		return exact_match, exact_matched_variation, None, min_distance, 0.0

	if exact_match:
		return exact_match, exact_matched_variation, exact_matched_variation, 0, 1.0

	if not captured_value or not accepted_answers:
		return exact_match, exact_matched_variation, None, None, None

	best_variation = min(
		(str(answer).strip() for answer in accepted_answers),
		key=lambda answer: levenshtein_distance(captured_value, answer),
	)
	min_distance = levenshtein_distance(captured_value, best_variation)
	max_length = max(len(captured_value), len(best_variation))
	normalised_similarity = 1.0 - (min_distance / max_length) if max_length else 1.0
	return exact_match, exact_matched_variation, best_variation, min_distance, normalised_similarity


def classify_risk(exact_match, normalised_similarity, cosine_similarity):
	if exact_match:
		return 'exact_match'
	if cosine_similarity is None:
		return 'check_for_spelling'
	if normalised_similarity is None:
		return 'unclassified'
	if (normalised_similarity < 0.8) != (cosine_similarity < 0.8):
		return 'high_risk'
	return 'low_risk'


def print_progress(current_row, total_rows):
	print(f'\rProcessing rows: {current_row}/{total_rows}\033[K', end='', flush=True)


def print_loading(message):
	print(f'\r{message}\033[K', end='', flush=True)


def run_pipeline(input_path, output_path, captured_column, answer_column, glove_model_name):
	data = read_csv_with_fallback_encodings(input_path)
	missing_columns = {captured_column, answer_column}.difference(data.columns)
	if missing_columns:
		raise ValueError(f'Input CSV is missing required columns: {sorted(missing_columns)}')

	model = None
	results = []
	for index, row in data.iterrows():
		captured_value = _text(row[captured_column])
		answer_key = row[answer_column]
		accepted_answers = expand_answer_variations(answer_key)
		is_multiple_choice = is_mcq_answer_key(answer_key)

		if is_multiple_choice:
			results.append({
				'exact_match': captured_value.casefold() == _text(answer_key).casefold(),
				'exact_matched_variation': _text(answer_key) if captured_value.casefold() == _text(answer_key).casefold() else None,
				'best_distance_variation': None,
				'min_distance': None,
				'normalised_similarity': None,
				'best_word2vec_variation': None,
				'max_cosine_similarity': None,
				'risk_class': classify_mcq(captured_value, answer_key),
			})
			print_progress(len(results), len(data))
			continue

		(
			exact_match,
			exact_matched_variation,
			best_distance_variation,
			min_distance,
			normalised_similarity,
		) = _distance_metrics(captured_value, accepted_answers, answer_key)

		if exact_match:
			best_word2vec_variation = exact_matched_variation
			max_cosine_similarity = 1.0
		elif captured_value.casefold() == '--blank--':
			best_word2vec_variation = None
			max_cosine_similarity = 0.0
		elif accepted_answers and captured_value:
			if model is None:
				print_loading(f'Loading pretrained GloVe model: {glove_model_name}')
				model = load_pretrained_glove_model(glove_model_name)
			best_word2vec_variation, max_cosine_similarity = best_word2vec_cosine_match(
				captured_value,
				accepted_answers,
				model,
			)
		else:
			best_word2vec_variation = None
			max_cosine_similarity = None

		results.append({
			'exact_match': exact_match,
			'exact_matched_variation': exact_matched_variation,
			'best_distance_variation': best_distance_variation,
			'min_distance': min_distance,
			'normalised_similarity': normalised_similarity,
			'best_word2vec_variation': best_word2vec_variation,
			'max_cosine_similarity': max_cosine_similarity,
			'risk_class': classify_risk(exact_match, normalised_similarity, max_cosine_similarity),
		})
		print_progress(len(results), len(data))

	print()
	result_data = pd.concat([data, pd.DataFrame(results, index=data.index)], axis=1)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	result_data.to_csv(output_path, index=False)
	print(f'Saved scored CSV to {output_path}')


def parse_args():
	parser = argparse.ArgumentParser(
		description='Score a HITL or no-human CSV with edit distance and Word2Vec similarity.'
	)
	parser.add_argument('input_csv', type=Path, help='Path to the raw HITL or no-human CSV.')
	parser.add_argument(
		'--output-csv',
		type=Path,
		help='Output path. Defaults to <input-name>_scored.csv next to the input.',
	)
	parser.add_argument('--captured-column', default='Captured')
	parser.add_argument('--answer-column', default='ANSWER Key')
	parser.add_argument('--glove-model', default='glove-wiki-gigaword-100')
	return parser.parse_args()


if __name__ == '__main__':
	arguments = parse_args()
	input_type = 'human' if arguments.input_csv.stem.casefold() == 'human' else 'no_human'
	output_csv = arguments.output_csv or (
		Path('output') / 'pipeline' / f'pipeline_{input_type}_{time.strftime("%Y%m%d_%H%M%S")}.csv'
	)
	run_pipeline(
		input_path=arguments.input_csv,
		output_path=output_csv,
		captured_column=arguments.captured_column,
		answer_column=arguments.answer_column,
		glove_model_name=arguments.glove_model,
	)
