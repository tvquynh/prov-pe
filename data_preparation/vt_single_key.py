#!/usr/bin/env python3
"""Optional future VT report retrieval using one API key and bounded quotas.

This client was not used to collect the published PROV-PE dataset.
It retrieves existing reports only and performs no sample submission or labeling.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

VERSION = "1.0.0"
BASE_URL = "https://www.virustotal.com/api/v3/files/"
MIN_INTERVAL = 16.0
MAX_DAILY = 500
MAX_MONTHLY = 15500
MAX_BODY = 16 * 1024 * 1024
SHA = re.compile(r"^[0-9a-f]{64}$")


def read_api_key(environ=None):
    value = (os.environ if environ is None else environ).get("VT_API_KEY", "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("Set VT_API_KEY to one valid 64-character hexadecimal API key.")
    return value


def read_hashes(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        values = pq.read_table(path, columns=["sha256"])["sha256"].to_pylist()
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if "sha256" not in (reader.fieldnames or []):
                raise ValueError("CSV input must contain a sha256 column.")
            values = [row["sha256"] for row in reader]
    else:
        values = path.read_text(encoding="utf-8-sig").splitlines()
    hashes, seen = [], set()
    for line, value in enumerate(values, 1):
        value = value.strip().lower() if isinstance(value, str) else ""
        if not SHA.fullmatch(value):
            raise ValueError(f"Invalid SHA-256 at input row {line}.")
        if value in seen:
            raise ValueError(f"Duplicate SHA-256 at input row {line}; prepare unique input.")
        seen.add(value)
        hashes.append(value)
    if not hashes:
        raise ValueError("Input contains no SHA-256 values.")
    return hashes


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class QuotaStop(Exception):
    pass


class Quota:
    """Persist reservations before requests; one state is shared across runs."""

    def __init__(self, path, api_key, daily_limit=MAX_DAILY, interval=MIN_INTERVAL):
        if not 1 <= daily_limit <= MAX_DAILY or not math.isfinite(interval) or interval < MIN_INTERVAL:
            raise ValueError("Daily limit must be 1..500 and interval at least 16 seconds.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), credential_digest TEXT NOT NULL, day TEXT NOT NULL, month TEXT NOT NULL, day_used INTEGER NOT NULL, month_used INTEGER NOT NULL, next_at REAL NOT NULL, blocked_until REAL NOT NULL)")
        fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO state VALUES(1,?,'','',0,0,0,0)", (fingerprint,))
        if self.db.execute("SELECT credential_digest FROM state WHERE id=1").fetchone()[0] != fingerprint:
            self.db.close()
            raise ValueError("The quota state belongs to a different credential; review the configuration.")
        self.daily_limit, self.interval = daily_limit, interval

    def reserve(self, now):
        utc = datetime.fromtimestamp(now, timezone.utc)
        day, month = utc.strftime("%Y-%m-%d"), utc.strftime("%Y-%m")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old_day, old_month, day_used, month_used, next_at, blocked = self.db.execute("SELECT day,month,day_used,month_used,next_at,blocked_until FROM state WHERE id=1").fetchone()
            day_used = day_used if old_day == day else 0
            month_used = month_used if old_month == month else 0
            if now < blocked:
                raise QuotaStop("Server cooldown remains active; retry later with the same state.")
            if day_used >= self.daily_limit or month_used >= MAX_MONTHLY:
                raise QuotaStop("Local daily or monthly allowance reached; no request sent.")
            if now < next_at:
                self.db.execute("COMMIT")
                return next_at - now
            self.db.execute("UPDATE state SET day=?,month=?,day_used=?,month_used=?,next_at=? WHERE id=1", (day, month, day_used + 1, month_used + 1, now + self.interval))
            self.db.execute("COMMIT")
            return 0.0
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def block(self, until):
        self.db.execute("UPDATE state SET blocked_until=MAX(blocked_until,?) WHERE id=1", (until,))

    def close(self):
        self.db.close()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def cooldown(header, now, fallback=3600):
    try:
        if str(header).strip().isdigit():
            return now + max(MIN_INTERVAL, int(header))
        value = parsedate_to_datetime(header)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return max(now + MIN_INTERVAL, value.timestamp())
    except (ValueError, TypeError, OverflowError):
        return now + fallback


def retrieve(sha256, api_key, opener):
    if not SHA.fullmatch(sha256):
        raise ValueError("Invalid SHA-256.")
    request = urllib.request.Request(BASE_URL + sha256, headers={"x-apikey": api_key, "Accept": "application/json", "User-Agent": "PROV-PE-single-key/" + VERSION})
    try:
        with opener.open(request, timeout=60) as response:
            body = response.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                return "invalid_response", response.status, None, None
            try:
                data = json.loads(body)["data"]
                valid = data["type"] == "file" and data["id"].lower() == sha256 and isinstance(data["attributes"], dict)
            except (ValueError, KeyError, TypeError, AttributeError):
                valid = False
            return ("ok" if valid and response.status == 200 else "invalid_response", response.status, body if valid else None, None)
    except urllib.error.HTTPError as exc:
        status = exc.code
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        exc.close()
        category = "not_found" if status == 404 else "rate_limit" if status == 429 else "authentication_error" if status in (401, 403) else "http_error"
        return category, status, None, retry_after
    except (urllib.error.URLError, TimeoutError, OSError):
        return "network_error", None, None, None


def run(hashes, output, api_key, quota, opener, max_samples, clock=time.time, sleep=time.sleep):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "reports.sqlite3")
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS observations (sha256 TEXT PRIMARY KEY, status TEXT, observed_at TEXT, http_status INTEGER, response_path TEXT, response_sha256 TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, sha256 TEXT, requested_at TEXT, status TEXT, http_status INTEGER)")
        identity = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
        previous = db.execute("SELECT value FROM metadata WHERE key='input_order_sha256'").fetchone()
        if previous and previous[0] != identity:
            raise ValueError("Output directory is bound to a different ordered input.")
        db.execute("INSERT OR IGNORE INTO metadata VALUES('input_order_sha256',?)", (identity,))
        db.execute("INSERT OR IGNORE INTO metadata VALUES('client_version',?)", (VERSION,))
        db.commit()
        attempted = 0
        for sha256 in hashes:
            if db.execute("SELECT 1 FROM observations WHERE sha256=?", (sha256,)).fetchone():
                continue
            if attempted >= max_samples:
                break
            while True:
                delay = quota.reserve(clock())
                if delay == 0:
                    break
                sleep(min(delay, 30))
            requested_at = datetime.fromtimestamp(clock(), timezone.utc).isoformat()
            cursor = db.execute("INSERT INTO attempts(sha256,requested_at,status) VALUES(?,?,'started')", (sha256, requested_at))
            attempt_id = cursor.lastrowid
            db.commit()
            status, http_status, body, retry_after = retrieve(sha256, api_key, opener)
            observed_at = datetime.fromtimestamp(clock(), timezone.utc).isoformat()
            attempted += 1
            db.execute("UPDATE attempts SET status=?,http_status=? WHERE id=?", (status, http_status, attempt_id))
            if status in ("ok", "not_found"):
                response_path, response_sha = None, None
                if body is not None:
                    folder = output / "responses";folder.mkdir(exist_ok=True)
                    path = folder / (sha256 + ".json.gz")
                    temp = path.with_suffix(".tmp")
                    with temp.open("wb") as stream:
                        stream.write(gzip.compress(body, mtime=0));stream.flush();os.fsync(stream.fileno())
                    os.replace(temp, path)
                    response_path, response_sha = path.relative_to(output).as_posix(), hashlib.sha256(body).hexdigest()
                db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?)", (sha256, status, observed_at, http_status, response_path, response_sha))
            db.commit()
            print(json.dumps({"sha256": sha256, "status": status, "http_status": http_status}), flush=True)
            if status == "rate_limit":
                quota.block(cooldown(retry_after, clock()))
                return 75
            if status == "authentication_error":
                return 77
            if status not in ("ok", "not_found"):
                quota.block(clock() + 60)
                return 75
        return 0
    finally:
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Unique SHA-256 input: TXT, CSV or Parquet with sha256 column")
    parser.add_argument("--output", type=Path, required=True, help="Private response directory; reuse it to resume")
    parser.add_argument("--state", type=Path, default=Path.home()/".prov-pe"/"vt_quota.sqlite3", help="Persistent shared quota state; preserve it across runs")
    parser.add_argument("--daily-limit", type=int, default=MAX_DAILY)
    parser.add_argument("--interval-seconds", type=float, default=MIN_INTERVAL)
    parser.add_argument("--max-samples", type=int, default=MAX_DAILY)
    parser.add_argument("--dry-run", action="store_true", help="Validate input and exit without credentials or network requests")
    args = parser.parse_args(argv)
    try:
        hashes = read_hashes(args.input)
        if not 1 <= args.max_samples <= MAX_DAILY:
            raise ValueError("max-samples must be 1..500.")
        if not 1 <= args.daily_limit <= MAX_DAILY or not math.isfinite(args.interval_seconds) or args.interval_seconds < MIN_INTERVAL:
            raise ValueError("Daily limit must be 1..500 and interval at least 16 seconds.")
        if args.dry_run:
            print(json.dumps({"status":"VALID_INPUT","unique_sha256":len(hashes),"network_requests":0}));return 0
        api_key = read_api_key()
        args.output.mkdir(parents=True, exist_ok=True)
        lock_path = args.output / ".retrieval.lock"
        try:
            lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise ValueError("Output is locked. Check for an active process before removing a stale lock.") from None
        quota = None
        try:
            os.write(lock, str(os.getpid()).encode());os.close(lock)
            quota = Quota(args.state, api_key, args.daily_limit, args.interval_seconds)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            return run(hashes, args.output, api_key, quota, opener, args.max_samples)
        finally:
            if quota is not None:quota.close()
            lock_path.unlink(missing_ok=True)
    except QuotaStop as exc:
        print(str(exc), file=sys.stderr);return 75
    except (ValueError, OSError, sqlite3.Error, ImportError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr);return 2
    except KeyboardInterrupt:
        print("Stopped; completed reports are preserved. Resume with the same input, output and quota state.", file=sys.stderr);return 130


if __name__ == "__main__":
    sys.exit(main())
