#!/usr/bin/env bash
# Ask the running vproc service a question and pretty-print the cited answer.
# Usage: scripts/ask.sh "What did we decide about the middleware?"
# Target host/port come from VPROC_ASK_URL or default to localhost:${VPROC_PORT:-8765}.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 \"<question>\"" >&2; exit 1; }
URL="${VPROC_ASK_URL:-http://localhost:${VPROC_PORT:-8765}}"
RESPONSE=$(curl -sS -m 600 -X POST "$URL/ask" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$1")" \
  -w '\n%{http_code}')
printf '%s' "$RESPONSE" | python3 -c '
import json, sys
body, _, status = sys.stdin.read().rpartition("\n")
try:
    a = json.loads(body)
except json.JSONDecodeError:
    sys.exit(f"vproc (HTTP {status}): invalid JSON response.")
if not status.startswith("2"):
    detail = a.get("detail") if isinstance(a, dict) else None
    if not isinstance(detail, str):
        detail = "Request failed."
    sys.exit(f"vproc (HTTP {status}): {detail}")
print("answered:", a["answered"], "| abstained:", a["abstained"])
print(a["text"])
for c in a.get("claims", []):
    cites = "; ".join("{} @ {}s ({})".format(x["memory_title"], int(x["start_ts"]), x["speaker"])
                      for x in c.get("citations", []))
    print("  - {}   [{}]".format(c["text"], cites))
'
