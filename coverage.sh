#!/usr/bin/env bash
#
# coverage.sh runs the tests and prints what they cover, per package.
#
# It writes coverage.txt. GOTESTFLAGS adds flags to `go test`, and
# GOEXPERIMENT=simd covers the vector kernels of the CPU it runs on. The
# exit status is go test's.

set -euo pipefail

cd "$(dirname "$0")"

# -covermode=set: atomic counters in the kernels' inner loops, which every
# thread runs at once, made the tests take over 10 minutes instead of 25 s.
# A covered or not is all a report shows.
#
# shellcheck disable=SC2086 # GOTESTFLAGS is a list of flags
go test -count=1 ${GOTESTFLAGS:-} \
    -covermode=set -coverpkg=./... -coverprofile=coverage-raw.txt ./...

# With -coverpkg, a block is listed once per test binary that links it. A
# block counts as covered when any test reached it.
{
    echo "mode: set"
    tail -n +2 coverage-raw.txt |
        awk '{ k = $1 " " $2; if ($3 > 0) count[k] = 1; else if (!(k in count)) count[k] = 0 } END { for (k in count) print k, count[k] }' |
        sort
} > coverage.txt
rm coverage-raw.txt

echo ""
echo "Statement coverage:"
awk '
    NR == 1 { next }
    {
        split($1, loc, ":")
        pkg = loc[1]
        sub(/\/[^\/]*$/, "", pkg)
        all[pkg] += $2
        total += $2
        if ($3 > 0) { hit[pkg] += $2; covered += $2 }
    }
    END {
        for (p in all) printf "  %-45s %6.1f%%  (%d/%d)\n", p, 100 * hit[p] / all[p], hit[p], all[p] | "sort"
        close("sort")
        printf "  %-45s %6.1f%%  (%d/%d)\n", "total", 100 * covered / total, covered, total
    }
' coverage.txt
