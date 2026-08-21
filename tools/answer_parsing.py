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


def _split_top_level_alternatives(text, preserve_whitespace=False):
	parts = []
	current = []
	depth = 0
	for char in text:
		if char in '[(':
			depth += 1
		elif char in '])':
			depth -= 1
		elif char == '/' and depth == 0:
			raw_part = ''.join(current)
			part = raw_part if preserve_whitespace else raw_part.strip()
			if part or (preserve_whitespace and raw_part):
				parts.append(part)
			current = []
			continue
		current.append(char)

	raw_part = ''.join(current)
	part = raw_part if preserve_whitespace else raw_part.strip()
	if part or (preserve_whitespace and raw_part):
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


def _take_unique(values, limit):
	"""Take at most ``limit`` values in grammar order, without an OCR reference."""
	result = []
	seen = set()
	for value in values:
		if value is None or value in seen:
			continue
		seen.add(value)
		result.append(value)
		if len(result) >= limit:
			break
	return result


def expand_answer_variations_bounded(answer_key, limit=100):
	"""Expand an answer grammar without materialising its full Cartesian product.

	Expansion is deliberately blind to the captured OCR answer. At every group
	boundary it retains at most ``limit`` variations in answer-key grammar order.
	This makes parsing deterministic, cacheable per answer key, and prevents large
	keys from creating millions of intermediate strings.
	"""
	if pd.isna(answer_key):
		return []
	text = str(answer_key).strip()
	if not text:
		return []
	limit = max(1, int(limit))

	def expand_fragment(fragment):
		alternatives = _split_top_level_alternatives(fragment, preserve_whitespace=True)
		all_values = []
		for alternative in alternatives:
			values = ['']
			idx = 0
			while idx < len(alternative):
				if alternative[idx] in '[(':
					opening = alternative[idx]
					closing = ']' if opening == '[' else ')'
					end = _find_matching(alternative, idx, opening, closing)
					group = expand_fragment(alternative[idx + 1:end])
					if opening == '(':
						group = [''] + group
					values = _take_unique(
						(prefix + option for prefix in values for option in group),
						limit,
					)
					idx = end + 1
				else:
					end = idx
					while end < len(alternative) and alternative[end] not in '[(':
						end += 1
					literal = alternative[idx:end]
					values = _take_unique((value + literal for value in values), limit)
					idx = end
			all_values.extend(values)
			all_values = _take_unique(all_values, limit)
		return all_values

	return _take_unique(expand_fragment(text), limit)
