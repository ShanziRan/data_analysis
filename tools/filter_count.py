import argparse

try:
    from tools.question_type import (
        filter_by_language_group,
        filter_by_part_number,
        filter_by_pos,
        read_csv_with_fallback,
    )
except ModuleNotFoundError:
    # Allows running this file directly (python tools/filter_count.py).
    from question_type import (
        filter_by_language_group,
        filter_by_part_number,
        filter_by_pos,
        read_csv_with_fallback,
    )


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description='Count teleforms (scan_id) and rows matching the same filters used in question_type.py.'
    )
    parser.add_argument('--input-csv', default='data/hitl/human.csv', help='Input CSV path')
    parser.add_argument('--part-number-csv', default='data/part_number.csv', help='Part number mapping CSV path')
    parser.add_argument('--start', type=int, default=0, help='Inclusive 0-based row start')
    parser.add_argument('--end', type=int, default=None, help='Exclusive 0-based row end')
    parser.add_argument('--language-groups', nargs='*', default=None, help='Optional language-group filter')
    parser.add_argument('--pos', nargs='*', default=None, help='Optional PoS filter, e.g. 822/03 D441/01')
    parser.add_argument('--part-numbers', nargs='*', default=None, help='Optional part-number filter, e.g. 1 2 3')
    parser.add_argument(
        '--include-unmapped',
        action='store_true',
        help='When --part-numbers is set, include rows with unknown inferred part number',
    )
    return parser


def main():
    args = _build_cli_parser().parse_args()

    df = read_csv_with_fallback(args.input_csv)
    df = df.iloc[args.start:args.end].copy()

    if args.language_groups:
        df = filter_by_language_group(df, args.language_groups)

    if args.pos:
        df = filter_by_pos(df, args.pos)

    if args.part_numbers:
        df = filter_by_part_number(
            df,
            part_numbers=args.part_numbers,
            include_unmapped=args.include_unmapped,
            part_number_csv_path=args.part_number_csv,
        )

    teleform_count = df['scan_id'].nunique() if 'scan_id' in df.columns else None

    print(f'Rows matched: {len(df)}')
    print(f'Teleforms matched (scan_id): {teleform_count if teleform_count is not None else "N/A (no scan_id column)"}')


if __name__ == '__main__':
    main()
