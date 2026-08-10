import re

import pandas as pd


MAX_VARIATIONS = 1000


class VariationLimitExceededError(ValueError):
	def __init__(self, message, partial_variations=None):
		super().__init__(message)
		self.partial_variations = partial_variations or []


def _raise_variation_limit(total_count, answer_text, partial_variations=None):
	raise VariationLimitExceededError(
		f'Variation limit exceeded ({total_count} > {MAX_VARIATIONS}) for answer key: {answer_text}',
		partial_variations=partial_variations,
	)


def is_mcq_answer_key(answer_key):
	if pd.isna(answer_key):
		return False

	text = str(answer_key).strip()
	return bool(re.fullmatch(r'[A-Z]', text))


def _find_matching(text, start, open_char, close_char):
	depth = 0
	for idx in range(start, len(text)):
		if text[idx] == open_char:
			depth += 1
		elif text[idx] == close_char:
			depth -= 1
			if depth == 0:
				return idx
	raise ValueError(f'Unbalanced group in answer key: {text}')


def _split_top_level_alternatives(text):
	parts = []
	current = []
	depth = 0
	for char in text:
		if char in '[(':
			depth += 1
		elif char in '])':
			depth -= 1
		elif char == '/' and depth == 0:
			part = ''.join(current).strip()
			if part:
				parts.append(part)
			current = []
			continue
		current.append(char)

	part = ''.join(current).strip()
	if part:
		parts.append(part)
	return parts


def _expand_plain_slash_list(text):
	parts = [part.strip() for part in text.split('/') if part.strip()]
	if not parts:
		return []

	starter_pattern = re.compile(r'^(does|do|did|can|could|will|would|is|are|was|were|has|have|had)\b', re.IGNORECASE)
	stitched = []
	idx = 0

	while idx < len(parts):
		current = parts[idx]
		next_part = parts[idx + 1] if idx + 1 < len(parts) else None

		# Handle split phrases like "does Sandra/sing well" by joining adjacent fragments.
		if (
			next_part
			and starter_pattern.match(current)
			and len(current.split()) <= 3
			and not starter_pattern.match(next_part)
		):
			stitched.append(f"{current} {next_part}".strip())
			idx += 2
			continue

		stitched.append(current)
		idx += 1

	unique = list(dict.fromkeys(stitched))
	if len(unique) > MAX_VARIATIONS:
		_raise_variation_limit(len(unique), text, partial_variations=unique[:MAX_VARIATIONS])

	return unique


def expand_answer_variations(answer_key):
	if pd.isna(answer_key):
		return []

	text = str(answer_key).strip()
	if not text:
		return []
	try:
		# Single-character answer keys such as A, B, C are already complete
		if re.fullmatch(r'^[A-Za-z0-9]+$', text):
			return [text]

		if '[' not in text and '(' not in text:
			if '/' in text:
				return _expand_plain_slash_list(text)
			return [text]

		def expand_sequence(seq):
			variants = ['']
			idx = 0
			while idx < len(seq):
				if seq[idx] == '[':
					end = _find_matching(seq, idx, '[', ']')
					inner = seq[idx + 1:end]
					group_variants = expand_answer_variations(inner)
					variants = [prefix + value for prefix in variants for value in group_variants]
					if len(variants) > MAX_VARIATIONS:
						_raise_variation_limit(len(variants), text, partial_variations=variants[:MAX_VARIATIONS])
					idx = end + 1
				elif seq[idx] == '(':
					end = _find_matching(seq, idx, '(', ')')
					inner = seq[idx + 1:end]
					group_variants = [''] + expand_answer_variations(inner)
					variants = [prefix + value for prefix in variants for value in group_variants]
					if len(variants) > MAX_VARIATIONS:
						_raise_variation_limit(len(variants), text, partial_variations=variants[:MAX_VARIATIONS])
					idx = end + 1
				else:
					end = idx
					while end < len(seq) and seq[end] not in '([':
						end += 1
					literal = seq[idx:end]
					if literal:
						variants = [prefix + literal for prefix in variants]
						if len(variants) > MAX_VARIATIONS:
							_raise_variation_limit(len(variants), text, partial_variations=variants[:MAX_VARIATIONS])
					idx = end

			return variants

		variants = []
		for part in _split_top_level_alternatives(text):
			part_variants = expand_sequence(part)
			variants.extend(part_variants)
			unique_variants = list(dict.fromkeys(v for v in variants if v))
			if len(unique_variants) > MAX_VARIATIONS:
				_raise_variation_limit(
					len(unique_variants),
					text,
					partial_variations=unique_variants[:MAX_VARIATIONS],
				)
			variants = unique_variants

		return variants
	except VariationLimitExceededError as error:
		print(f'[answer_parsing] {error}. Truncating to {MAX_VARIATIONS} variations and continuing.')
		if error.partial_variations:
			return list(dict.fromkeys(v for v in error.partial_variations if v))[:MAX_VARIATIONS]
		return [text]
