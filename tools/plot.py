import math
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
	parser = argparse.ArgumentParser(description='Plot edit-distance normalised similarity distribution.')
	parser.add_argument(
		'--csv',
		default='output/distance/new_data/distance_35977_334034_human_exact_match.csv',
		help='Path to the edit-distance CSV file.',
	)
	args = parser.parse_args()

	csv_path = Path(args.csv)
	if not csv_path.exists():
		raise FileNotFoundError(f'CSV file not found: {csv_path}')

	output_dir = csv_path.parent
	output_dir.mkdir(parents=True, exist_ok=True)
	stem = csv_path.stem
	output_image_path = output_dir / f'normalised_similarity_distribution_{stem}.png'
	output_image_path_no_extremes = output_dir / f'normalised_similarity_distribution_no_extremes_{stem}.png'

	df = pd.read_csv(csv_path, encoding='utf-8')
	if 'normalised_similarity' not in df.columns:
		raise KeyError(
			"Column 'normalised_similarity' not found in the edit-distance CSV. "
			f'Available columns: {list(df.columns)}'
		)

	scores = pd.to_numeric(df['normalised_similarity'], errors='coerce').dropna()
	if scores.empty:
		raise ValueError('No valid normalised similarity scores to plot.')

	print(f'Loaded CSV: {csv_path}')
	print(f'Total rows: {len(df)}, rows with valid normalised_similarity: {len(scores)}')

	def _plot_scores(values, output_path, title_suffix, fit_curve=False):
		if values.empty:
			print(f'No values available for {title_suffix}; skipped plot.')
			return

		mean_value = values.mean()
		median_value = values.median()

		if fit_curve:
			std_value = values.std(ddof=1)
			x_values = np.linspace(float(values.min()), float(values.max()), 500)
			if std_value > 0:
				normal_pdf = (
					1.0 / (std_value * math.sqrt(2.0 * math.pi))
				) * np.exp(-0.5 * ((x_values - mean_value) / std_value) ** 2)
			else:
				normal_pdf = np.zeros_like(x_values)

		plt.figure(figsize=(10, 6))
		plt.hist(values, bins=100, edgecolor='black', alpha=0.75, label='Observed')
		if fit_curve:
			plt.plot(
				x_values,
				normal_pdf,
				color='navy',
				linewidth=2.0,
				label=f'Normal fit (mu={mean_value:.3f}, sigma={std_value:.3f})',
			)
		plt.axvline(mean_value, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_value:.3f}')
		plt.axvline(median_value, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_value:.3f}')
		plt.title(f'Edit-distance normalised similarity {title_suffix}')
		plt.xlabel('Normalised similarity')
		plt.ylabel('Frequency')
		plt.legend()
		plt.tight_layout()
		plt.savefig(output_path, dpi=150)
		plt.close()

	_plot_scores(scores, output_image_path, '(all values)', fit_curve=False)

	non_extreme_scores = scores[(scores > 0) & (scores < 1)]
	_plot_scores(non_extreme_scores, output_image_path_no_extremes, '(excluding 0 and 1)', fit_curve=True)

	print(f'Saved plot to {output_image_path}')
	print(f'Saved plot to {output_image_path_no_extremes}')


if __name__ == '__main__':
	main()

