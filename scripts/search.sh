#!/usr/bin/env bash
# Search the running vproc service (raw retrieval, no grounding) and print evidence chunks.
# Usage: scripts/search.sh "BILLINGCODE configuration"
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 \"<query>\"" >&2; exit 1; }
URL="${VPROC_ASK_URL:-http://localhost:${VPROC_PORT:-8765}}"
RESPONSE=$(curl -sS -m 120 -X POST "$URL/search" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1]}))' "$1")" \
  -w '\n%{http_code}')
printf '%s' "$RESPONSE" | python3 -c '
import json, sys
body, _, status = sys.stdin.read().rpartition("\n")
try:
    evidence = json.loads(body)
except json.JSONDecodeError:
    sys.exit(f"vproc (HTTP {status}): invalid JSON response.")
if not status.startswith("2"):
    detail = evidence.get("detail") if isinstance(evidence, dict) else None
    if not isinstance(detail, str):
        detail = "Request failed."
    sys.exit(f"vproc (HTTP {status}): {detail}")
for i, e in enumerate(evidence, 1):
    print("E{} [{}-{}s {}] {!r}".format(i, int(e["start_ts"]), int(e["end_ts"]),
                                        e["speaker"], e["text"][:140]))
'
