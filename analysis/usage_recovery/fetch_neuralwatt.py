"""Download Neuralwatt's per-request usage rows (content-free) for a time window.

    uv run python analysis/usage_recovery/fetch_neuralwatt.py 2026-10-09T00:00:00Z 2026-10-13T09:10:00Z neuralwatt-rows.jsonl.gz

Writes results/recovered-usage/records/<out> (gzip). API: GET /v1/usage/requests (30-day window, 200 rows per page,
20 requests per minute). Rows carry token counts, finish_reason, duration, energy and cost, but no
prompt or answer text; match.py ties them to questions.
"""

import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]


def main():
    start, end, name = sys.argv[1:4]
    key = dotenv_values(ROOT / ".env")["NEURALWATT_API_KEY"]
    out = ROOT / "results/recovered-usage/records" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    params, cursor, n = {"start_date": start, "end_date": end, "limit": 200}, None, 0
    with gzip.open(out, "wt", encoding="utf-8") as stream:
        while True:
            query = urllib.parse.urlencode({**params, **({"cursor": cursor} if cursor else {})})
            request = urllib.request.Request(f"https://api.neuralwatt.com/v1/usage/requests?{query}",
                                             headers={"Authorization": f"Bearer {key}"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500):
                    time.sleep(int(e.headers.get("Retry-After") or 30))
                    continue
                raise
            for row in body["requests"]:
                stream.write(json.dumps(row) + "\n")
            n += len(body["requests"])
            if not body["has_more"]:
                break
            cursor = body["next_cursor"]
            time.sleep(3.2)  # 20 requests per minute
    print(f"{n} rows -> {out} ({body['period']})")


if __name__ == "__main__":
    main()
