import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _default_csv_for_approach(approach: str) -> Path:
	approach = approach.casefold()
	if approach == 'tfidf':
		return Path('output/vector/new_data/tfidf_39027_145968_no_human.csv')
	if approach == 'word2vec':
		return Path('output/vector/word2vec_analysis_72678_462182_no_human_new.csv')
	raise ValueError("approach must be either 'tfidf' or 'word2vec'")


def _plot_scores(values, output_path, title, xlabel, fit_curve=False):
	if values.empty:
		print(f'No values available for {title}; skipped plot.')
		return

	mean_value = values.mean()
	median_value = values.median()

	plt.figure(figsize=(10, 6))
	plt.hist(values, bins=100, edgecolor='black', alpha=0.75, label='Observed')

	if fit_curve:
		std_value = values.std(ddof=1)
		x_values = np.linspace(float(values.min()), float(values.max()), 500)
		if std_value > 0:
			normal_pdf = (
				1.0 / (std_value * math.sqrt(2.0 * math.pi))
			) * np.exp(-0.5 * ((x_values - mean_value) / std_value) ** 2)
		else:
			normal_pdf = np.zeros_like(x_values)

		plt.plot(
			x_values,
			normal_pdf,
			color='navy',
			linewidth=2.0,
			label=f'Normal fit (mu={mean_value:.3f}, sigma={std_value:.3f})',
		)

	plt.axvline(mean_value, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_value:.3f}')
	plt.axvline(median_value, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_value:.3f}')
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('Frequency')
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def main():
	parser = argparse.ArgumentParser(description='Plot TF-IDF and Word2Vec similarity distributions.')
	parser.add_argument(
		'--approach',
		choices=('tfidf', 'word2vec'),
		default='tfidf',
		help='Vector approach to plot when --csv is not provided.',
	)
	parser.add_argument(
		'--csv',
		default=None,
		help='Path to the vector-based CSV file. If omitted, a default path is chosen from --approach.',
	)
	parser.add_argument(
		'--non-mcq-only',
		action='store_true',
		help='Plot only rows where multiple_choice is false.',
	)
	args = parser.parse_args()

	csv_path = Path(args.csv) if args.csv else _default_csv_for_approach(args.approach)
	if not csv_path.exists():
		raise FileNotFoundError(f'CSV file not found: {csv_path}')

	approach = args.approach.casefold()
	output_dir = csv_path.parent
	output_dir.mkdir(parents=True, exist_ok=True)
	stem = csv_path.stem
	file_suffix = '_non_mcq_only' if args.non_mcq_only else ''
	output_image_path = output_dir / f'max_cosine_similarity_distribution_{stem}{file_suffix}.png'
	output_image_path_no_extremes = output_dir / f'max_cosine_similarity_distribution_no_extremes_{stem}{file_suffix}.png'

	df = pd.read_csv(csv_path, encoding='utf-8')
	if 'max_cosine_similarity' not in df.columns:
		raise KeyError(
			"Column 'max_cosine_similarity' not found in the vector CSV. "
			f'Available columns: {list(df.columns)}'
		)
	if args.non_mcq_only:
		if 'multiple_choice' not in df.columns:
			raise KeyError(
				"Column 'multiple_choice' not found in the vector CSV, so non-MCQ filtering can't be applied. "
				f'Available columns: {list(df.columns)}'
			)
		multiple_choice = pd.Series(df['multiple_choice']).astype(str).str.casefold()
		non_mcq_mask = multiple_choice.isin(('false', '0', 'nan', 'none', ''))
		df = df.loc[non_mcq_mask].copy()

	scores = pd.to_numeric(df['max_cosine_similarity'], errors='coerce').dropna()
	if scores.empty:
		raise ValueError('No valid max_cosine_similarity scores to plot.')

	print(f'Loaded CSV: {csv_path}')
	print(f'Approach: {approach}')
	print(f'Non-MCQ only: {args.non_mcq_only}')
	print(f'Total rows: {len(df)}, rows with valid max_cosine_similarity: {len(scores)}')

	title_prefix = 'TF-IDF' if approach == 'tfidf' else 'Word2Vec'
	_plot_scores(
		scores,
		output_image_path,
		f'{title_prefix} max cosine similarity (all values)',
		'Max cosine similarity',
		fit_curve=False,
	)

	non_extreme_scores = scores[(scores > 0) & (scores < 1)]
	_plot_scores(
		non_extreme_scores,
		output_image_path_no_extremes,
		f'{title_prefix} max cosine similarity (excluding 0 and 1)',
		'Max cosine similarity',
		fit_curve=True,
	)

	print(f'Saved plot to {output_image_path}')
	print(f'Saved plot to {output_image_path_no_extremes}')


if __name__ == '__main__':
	main()
