import argparse
import json
from collections import Counter


def _normalise_text(value):
    return str(value).strip().casefold()


def extract_character_changes(source, target):
    source_text = _normalise_text(source)
    target_text = _normalise_text(target)

    if source_text == target_text:
        return []

    rows = len(source_text) + 1
    cols = len(target_text) + 1
    matrix = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        matrix[i][0] = i
    for j in range(cols):
        matrix[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = 0 if source_text[i - 1] == target_text[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + substitution_cost,
            )

    changes = []
    i = len(source_text)
    j = len(target_text)

    while i > 0 or j > 0:
        if i > 0 and j > 0 and source_text[i - 1] == target_text[j - 1] and matrix[i][j] == matrix[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue

        if i > 0 and j > 0 and matrix[i][j] == matrix[i - 1][j - 1] + 1:
            changes.append(f"{source_text[i - 1]} -> {target_text[j - 1]}")
            i -= 1
            j -= 1
            continue

        if i > 0 and matrix[i][j] == matrix[i - 1][j] + 1:
            changes.append(f"{source_text[i - 1]} -> <del>")
            i -= 1
            continue

        if j > 0 and matrix[i][j] == matrix[i][j - 1] + 1:
            changes.append(f"<ins> -> {target_text[j - 1]}")
            j -= 1
            continue

        # Safety fallback for unexpected ties.
        if i > 0 and j > 0:
            changes.append(f"{source_text[i - 1]} -> {target_text[j - 1]}")
            i -= 1
            j -= 1
        elif i > 0:
            changes.append(f"{source_text[i - 1]} -> <del>")
            i -= 1
        else:
            changes.append(f"<ins> -> {target_text[j - 1]}")
            j -= 1

    changes.reverse()
    return changes


def summarise_character_changes(changes):
    return dict(sorted(Counter(changes).items(), key=lambda item: (-item[1], item[0])))


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description='Show character-level changes from captured text to best answer variation.'
    )
    parser.add_argument('captured', help='Captured value')
    parser.add_argument('best_variation', help='Best answer key variation')
    return parser


def main():
    args = _build_cli_parser().parse_args()
    changes = extract_character_changes(args.captured, args.best_variation)
    payload = {
        'captured': args.captured,
        'best_variation': args.best_variation,
        'changes': changes,
        'change_counts': summarise_character_changes(changes),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
