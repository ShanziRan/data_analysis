import json
import re
import time
from pathlib import Path

import pandas as pd

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

    return list(dict.fromkeys(stitched))


def expand_answer_variations(answer_key):
    if pd.isna(answer_key):
        return []

    text = str(answer_key).strip()
    if not text:
        return []

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
                idx = end + 1
            elif seq[idx] == '(':
                end = _find_matching(seq, idx, '(', ')')
                inner = seq[idx + 1:end]
                group_variants = [''] + expand_answer_variations(inner)
                variants = [prefix + value for prefix in variants for value in group_variants]
                idx = end + 1
            else:
                end = idx
                while end < len(seq) and seq[end] not in '([':
                    end += 1
                literal = seq[idx:end]
                if literal:
                    variants = [prefix + literal for prefix in variants]
                idx = end

        return variants

    variants = []
    for part in _split_top_level_alternatives(text):
        variants.extend(expand_sequence(part))

    return list(dict.fromkeys(v for v in variants if v))


def levenshtein_distance(source, target):
    # Normalise the input strings by stripping whitespace and converting to lowercase
    source = str(source).strip().casefold()
    target = str(target).strip().casefold()

    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current_row = [i]
        for j, target_char in enumerate(target, start=1):
            insertion_cost = current_row[j - 1] + 1
            deletion_cost = previous_row[j] + 1
            substitution_cost = previous_row[j - 1] + (0 if source_char == target_char else 1)
            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
        previous_row = current_row

    return previous_row[-1]


def damerau_levenshtein_distance(source, target):
    # Normalise the input strings by stripping whitespace and converting to lowercase
    source = str(source).strip().casefold()
    target = str(target).strip().casefold()

    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    len_source = len(source)
    len_target = len(target)
    max_dist = len_source + len_target

    matrix = [[0] * (len_target + 2) for _ in range(len_source + 2)]
    matrix[0][0] = max_dist

    for i in range(len_source + 1):
        matrix[i + 1][0] = max_dist
        matrix[i + 1][1] = i

    for j in range(len_target + 1):
        matrix[0][j + 1] = max_dist
        matrix[1][j + 1] = j

    last_row = {}

    for i in range(1, len_source + 1):
        last_match_col = 0
        for j in range(1, len_target + 1):
            i1 = last_row.get(target[j - 1], 0)
            j1 = last_match_col
            cost = 0 if source[i - 1] == target[j - 1] else 1

            if source[i - 1] == target[j - 1]:
                last_match_col = j
                matrix[i + 1][j + 1] = min(
                    matrix[i][j] + cost,
                    matrix[i + 1][j] + 1,
                    matrix[i][j + 1] + 1,
                    matrix[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
                )
            else:
                matrix[i + 1][j + 1] = min(
                    matrix[i][j] + cost,
                    matrix[i + 1][j] + 1,
                    matrix[i][j + 1] + 1,
                    matrix[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
                )

        last_row[source[i - 1]] = i

    return matrix[len_source + 1][len_target + 1]


def run_distance_analysis(row_start, row_end, skipped_rows=None):
    output_log = []
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'distance_analysis_{row_start}_{row_end}_ref.json'

    skip_set = set(skipped_rows or [])

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting answer key processing")
    answer_key_start = time.perf_counter()

    data['distance'] = pd.NA
    data['best_variation'] = pd.NA
    data['similarity'] = pd.NA

    selected_answers = data['ANSWER Key'].iloc[row_start:row_end].dropna()
    selected_answers = selected_answers.drop(index=skip_set, errors='ignore')

    expanded_variations = {}
    for idx, answer in selected_answers.items():
        accepted_answers = expand_answer_variations(answer)
        expanded_variations[idx] = accepted_answers
        message = f"Row {idx}: {len(accepted_answers)} variations"
        print(message)

    answer_key_elapsed = time.perf_counter() - answer_key_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished answer key processing in {answer_key_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting minimum distance calculation")
    distance_start = time.perf_counter()

    for idx, accepted_answers in expanded_variations.items():
        captured_value = str(data.loc[idx, 'Captured']).strip()
        if not accepted_answers or not captured_value:
            min_distance = None
            best_variation = None
        else:
            min_distance = None
            best_variation = None
            for candidate in accepted_answers:
                candidate_text = str(candidate).strip()
                distance = levenshtein_distance(captured_value, candidate_text)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    best_variation = candidate_text

        if min_distance is None or not captured_value:
            similarity = None
        else:
            max_len = max(len(captured_value), len(best_variation or ""))
            # similarity = 1.0 - (min_distance / max_len) if max_len else 1.0
            similarity = 1.0 - (min_distance / len(best_variation)) if best_variation else 1.0

        data.at[idx, 'distance'] = min_distance
        data.at[idx, 'best_variation'] = best_variation
        data.at[idx, 'similarity'] = similarity
        uid_value = data.loc[idx, 'UID']
        confidence_value = data.loc[idx, 'confidence']
        message = (
            f"Row {idx+2}: UID = {uid_value}, confidence = {confidence_value}, "
            f"captured = {captured_value}, min distance = {min_distance}, "
            f"best variation = {best_variation}, similarity = {similarity}"
        )
        print(message)
        output_log.append({
            'row': idx+2,
            'UID': uid_value,
            'confidence': confidence_value,
            'captured': captured_value,
            'min_distance': min_distance,
            'best_variation': best_variation,
            'similarity': similarity,
        })

    distance_elapsed = time.perf_counter() - distance_start
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished minimum distance calculation in {distance_elapsed:.2f} seconds"
    print(message)
    output_log.append({'type': 'info', 'message': message})

    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(output_log, handle, indent=2, ensure_ascii=False)

    print(f"Saved output log to {output_path}")


# Set row range here (0-indexed) and specify any rows to skip (0-indexed)
ROW_START = 231636
ROW_END = 231688
SKIP_ROWS = []

run_distance_analysis(row_start=ROW_START, row_end=ROW_END, skipped_rows=SKIP_ROWS)
