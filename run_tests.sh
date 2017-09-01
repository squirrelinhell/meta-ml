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

for test in $(find tests -name '*.py' | sort); do
    echo -n "Test: $test... "

    OUT_FILE="${test%.*}.out"
    if [ -f "$OUT_FILE" ]; then
        cat "$OUT_FILE"
    fi > "$TMPDIR/ans"

    python3 "$test" </dev/null >"$TMPDIR/out" 2>"$TMPDIR/dbg"
    RESULT=$?

    if ! [ "x$RESULT" = x0 ]; then
        echo FAIL
        cat "$TMPDIR/dbg"
        echo
        echo "EXIT CODE $RESULT: $test"
        echo
    elif ! diff -b -q "$TMPDIR/ans" "$TMPDIR/out" >/dev/null; then
        echo FAIL
        cat "$TMPDIR/dbg"
        echo
        echo "INCORRECT OUTPUT: $test"
        echo
        diff -b --color=auto "$TMPDIR/ans" "$TMPDIR/out"
        echo
    else
        echo OK
    fi
done
