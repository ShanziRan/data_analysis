#!/usr/bin/env bash

ROW_START=0
ROW_END=-1
SKIP_ROWS=""
PART_NUMBERS="2"
POS_VALUES="D441/03"
PART_NUMBER_CSV="data/part_number.csv"
NON_MCQ_ONLY=true
CHECK_HUMAN=false

ARGS=(
	--row-start "$ROW_START"
	--row-end "$ROW_END"
	--part-number-csv "$PART_NUMBER_CSV"
)

if [[ -n "$SKIP_ROWS" ]]; then
	ARGS+=(--skip-rows "$SKIP_ROWS")
fi

if [[ -n "$PART_NUMBERS" ]]; then
	ARGS+=(--part-numbers "$PART_NUMBERS")
fi

if [[ -n "$POS_VALUES" ]]; then
	ARGS+=(--pos "$POS_VALUES")
fi

if [[ "$NON_MCQ_ONLY" == true ]]; then
	ARGS+=(--non-mcq-only)
fi

if [[ "$CHECK_HUMAN" == true ]]; then
	ARGS+=(--check-human)
fi

python edit_distance.py "${ARGS[@]}"

