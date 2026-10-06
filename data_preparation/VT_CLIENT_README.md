# Optional VirusTotal report client

`vt_single_key.py` uses one API key from `VT_API_KEY` to retrieve existing reports by SHA-256. It never submits files, requests rescans, trains models or assigns labels. This client is supplied for future use; it was not used to collect the published PROV-PE dataset. Reproducing the paper uses the frozen released metadata and saved outputs, not fresh API queries.

The client sends requests sequentially, with at least 16 seconds between requests, at most 500 attempts per UTC day and at most 15,500 per UTC month. Daily and interval options may only reduce throughput. All attempted requests count against the local allowance. Server limits and the account's actual remaining allowance take precedence. API use elsewhere also consumes the account allowance, so reduce the local daily budget accordingly.

On HTTP 429 the client records the server's `Retry-After` cooldown, or one hour if the header is absent, and stops. It does not automatically retry. HTTP 401/403 also stops immediately. Other HTTP, network and response-validation failures stop with a one-minute local cooldown. None of these failures is marked as a completed report. HTTP 404 is recorded explicitly as not found. Resume with the same input, output and quota state after resolving the cause. A successful run may stop after `--max-samples`; inspect `reports.sqlite3` for progress.

Requirements: Python 3.10 or newer. TXT and CSV input use only the standard library. Parquet input also requires `pyarrow`. Input must contain unique, valid SHA-256 values; CSV and Parquet use a `sha256` column. A TXT file contains one digest per line.

Validate without contacting VirusTotal:

```powershell
python data_preparation/vt_single_key.py --input hashes.txt --output private_vt_reports --dry-run
```

Provide the API key through a private environment or secret manager as `VT_API_KEY`, then run:

```powershell
python data_preparation/vt_single_key.py --input hashes.txt --output private_vt_reports
```

The default quota database is `~/.prov-pe/vt_quota.sqlite3`. Preserve this state across restarts and share it across uses of the client. Deleting it or choosing a new state path does not restore the server allowance. One process may write an output directory at a time. After an abnormal termination, inspect the process status before removing a stale `.retrieval.lock`; preserve the quota database.

The private output contains raw JSON responses compressed with gzip and a SQLite ledger recording request time, observation time, HTTP state and response hash. Do not publish API credentials or private raw reports. Fresh report contents and timestamps may differ from the frozen dataset.

Tests use synthetic hashes and mocked responses only:

```powershell
python -m unittest discover -s data_preparation/tests -v
```

Use requires an authorized VirusTotal account and compliance with its terms. See the [official API limits](https://docs.virustotal.com/reference/public-vs-premium-api) and [quota documentation](https://docs.virustotal.com/docs/quota-consumption).
