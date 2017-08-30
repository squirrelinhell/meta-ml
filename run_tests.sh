#!/bin/bash

TMPDIR=$(mktemp -d) || exit 1
trap "rm -rf $TMPDIR" EXIT

export PYTHONPATH="$(pwd):$PYTHONPATH"

if [ "x$1" != x ]; then
    TEST_FILE="tests/$1"
    [ -f "$TEST_FILE" ] || TEST_FILE="$TEST_FILE.py"
    export DEBUG=1
    echo "import debug" > "$TMPDIR/run.py"
    cat "$TEST_FILE" >> "$TMPDIR/run.py"
    exec python3 "$TMPDIR/run.py"
fi

for test in $(ls tests | grep 'py$'); do
    echo "Test: $test..."
    TEST_FILE="tests/$test"

    OUT_FILE="${TEST_FILE%.*}.out"
    if [ -f "$OUT_FILE" ]; then
        cat "$OUT_FILE"
    fi > "$TMPDIR/ans"

    if ! python3 "$TEST_FILE" </dev/null >"$TMPDIR/out" 2>/dev/null; then
        echo
        echo "TEST FAILED: $test"
        echo
    elif ! diff -b -q "$TMPDIR/ans" "$TMPDIR/out" >/dev/null; then
        echo
        echo "INCORRECT OUTPUT: $test"
        echo
        diff -b --color=auto "$TMPDIR/ans" "$TMPDIR/out"
    fi
done
