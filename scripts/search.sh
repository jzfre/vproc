#!/usr/bin/env bash
# Search the running vproc service (raw retrieval, no grounding) and print evidence chunks.
# Usage: scripts/search.sh "BILLINGCODE configuration"
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 \"<query>\"" >&2; exit 1; }
URL="${VPROC_ASK_URL:-http://localhost:${VPROC_PORT:-8765}}"
curl -sS -m 120 -X POST "$URL/search" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1]}))' "$1")" \
| python3 -c '
import json, sys
for i, e in enumerate(json.load(sys.stdin), 1):
    print("E{} [{}-{}s {}] {!r}".format(i, int(e["start_ts"]), int(e["end_ts"]),
                                        e["speaker"], e["text"][:140]))
'
