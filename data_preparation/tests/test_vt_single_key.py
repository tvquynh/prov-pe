"""Offline contract tests; no live VirusTotal requests."""
import contextlib
from datetime import datetime, timezone
from email.message import Message
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "vt_single_key.py"
spec = importlib.util.spec_from_file_location("vt_single_key", MODULE)
vt = importlib.util.module_from_spec(spec);spec.loader.exec_module(vt)
KEY = "a" * 64
H1, H2 = "1" * 64, "2" * 64


class Clock:
    def __init__(self):self.value = datetime(2026, 10, 6, tzinfo=timezone.utc).timestamp()
    def now(self):return self.value
    def sleep(self, seconds):self.value += seconds


class Response:
    status = 200
    def __init__(self, body):self.body = body
    def read(self, size):return self.body[:size]
    def __enter__(self):return self
    def __exit__(self, *args):return False


class Opener:
    def __init__(self, values):self.values=list(values);self.requests=[]
    def open(self, request, timeout):
        self.requests.append(request)
        value=self.values.pop(0)
        if isinstance(value, BaseException):raise value
        return Response(value)


def body(sha):
    return json.dumps({"data":{"type":"file","id":sha,"attributes":{"last_analysis_stats":{"malicious":0}}}}).encode()


def http_error(code, retry=None):
    headers=Message()
    if retry is not None:headers["Retry-After"]=retry
    return urllib.error.HTTPError(vt.BASE_URL+H1,code,"synthetic",headers,io.BytesIO(b""))


class TestClient(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.clock=Clock()
        self.quota=vt.Quota(self.root/'quota.sqlite3',KEY)
    def tearDown(self):self.quota.close();self.temp.cleanup()
    def execute(self, values, hashes=(H1,H2), limit=500):
        opener=Opener(values)
        with contextlib.redirect_stdout(io.StringIO()):
            result=vt.run(list(hashes),self.root/'out',KEY,self.quota,opener,limit,self.clock.now,self.clock.sleep)
        return result,opener
    def observations(self):
        with contextlib.closing(sqlite3.connect(self.root/'out/reports.sqlite3')) as db:
            return db.execute('SELECT sha256,status FROM observations ORDER BY sha256').fetchall()
    def test_key_is_required_and_not_printed(self):
        for value in ['',KEY+'\n'+KEY,KEY+','+KEY,'bad']:
            with self.assertRaises(ValueError) as error:vt.read_api_key({'VT_API_KEY':value})
            self.assertNotIn(KEY,str(error.exception))
        self.assertEqual(vt.read_api_key({'VT_API_KEY':KEY}),KEY)
    def test_input_validation_before_network(self):
        p=self.root/'hashes.txt';p.write_text(H1+'\n'+H2+'\n')
        self.assertEqual(vt.read_hashes(p),[H1,H2])
        p.write_text(H1+'\n'+H1+'\n')
        with self.assertRaises(ValueError):vt.read_hashes(p)
        p.write_text(H1+'\nwrong')
        with self.assertRaises(ValueError):vt.read_hashes(p)
    def test_csv_input(self):
        p=self.root/'hashes.csv';p.write_text('sha256,other\n'+H1+',x\n')
        self.assertEqual(vt.read_hashes(p),[H1])
    def test_dry_run_needs_no_key_and_sends_no_request(self):
        p=self.root/'hashes.txt';p.write_text(H1+'\n')
        with patch.object(vt.urllib.request,'build_opener',side_effect=AssertionError('network forbidden')),patch.dict('os.environ',{'VT_API_KEY':''}),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vt.main(['--input',str(p),'--output',str(self.root/'dry'),'--dry-run']),0)
        self.assertFalse((self.root/'dry').exists())
    def test_quota_persists_across_restarts(self):
        self.assertEqual(self.quota.reserve(self.clock.now()),0)
        self.quota.close();self.quota=vt.Quota(self.root/'quota.sqlite3',KEY)
        self.assertEqual(self.quota.reserve(self.clock.now()),16)
        self.clock.sleep(16);self.assertEqual(self.quota.reserve(self.clock.now()),0)
    def test_daily_and_monthly_caps(self):
        self.quota.reserve(self.clock.now())
        self.quota.db.execute('UPDATE state SET day_used=500')
        self.clock.sleep(60)
        with self.assertRaises(vt.QuotaStop):self.quota.reserve(self.clock.now())
        self.clock.sleep(86400);self.assertEqual(self.quota.reserve(self.clock.now()),0)
        self.quota.db.execute('UPDATE state SET month_used=15500')
        self.clock.sleep(60)
        with self.assertRaises(vt.QuotaStop):self.quota.reserve(self.clock.now())
    def test_configuration_cannot_increase_limits(self):
        for daily,interval in [(501,16),(500,15),(0,16),(500,float('nan'))]:
            with self.assertRaises(ValueError):vt.Quota(self.root/'bad.sqlite3',KEY,daily,interval)
    def test_success_preserves_response_and_resumes_without_request(self):
        start=self.clock.now();result,opener=self.execute([body(H1),body(H2)])
        self.assertEqual(result,0);self.assertEqual(len(opener.requests),2)
        self.assertGreaterEqual(self.clock.now()-start,16)
        self.assertEqual(gzip.decompress((self.root/'out/responses'/f'{H1}.json.gz').read_bytes()),body(H1))
        self.assertEqual(opener.requests[0].get_header('X-apikey'),KEY)
        result,opener=self.execute([]);self.assertEqual(result,0);self.assertEqual(len(opener.requests),0)
    def test_404_is_recorded_without_hiding_sample(self):
        result,opener=self.execute([http_error(404)],hashes=[H1])
        self.assertEqual(result,0);self.assertEqual(self.observations(),[(H1,'not_found')])
    def test_429_stops_and_persists_cooldown(self):
        now=self.clock.now();result,opener=self.execute([http_error(429,'7200'),body(H2)])
        self.assertEqual(result,75);self.assertEqual(len(opener.requests),1);self.assertEqual(self.observations(),[])
        self.quota.close();self.quota=vt.Quota(self.root/'quota.sqlite3',KEY)
        with self.assertRaises(vt.QuotaStop):self.quota.reserve(now+3600)
        self.clock.sleep(7200);self.assertEqual(self.quota.reserve(self.clock.now()),0)
    def test_retry_after_date_and_fallback(self):
        now=self.clock.now()
        self.assertEqual(vt.cooldown(None,now),now+3600)
        self.assertEqual(vt.cooldown('Tue, 06 Oct 2026 01:00:00 GMT',now),now+3600)
    def test_authentication_error_is_not_retried(self):
        for code in [401,403]:
            result,opener=self.execute([http_error(code),body(H2)])
            self.assertEqual(result,77);self.assertEqual(len(opener.requests),1);self.assertEqual(self.observations(),[])
    def test_server_and_network_failure_do_not_complete_sample(self):
        result,opener=self.execute([http_error(503),body(H2)])
        self.assertEqual(result,75);self.assertEqual(len(opener.requests),1);self.assertEqual(self.observations(),[])
        self.clock.sleep(61)
        result,opener=self.execute([urllib.error.URLError('synthetic timeout')])
        self.assertEqual(result,75);self.assertEqual(self.observations(),[])
    def test_malformed_or_wrong_identity_is_rejected(self):
        for value in [b'not-json',body(H2)]:
            self.clock.sleep(61)
            result,opener=self.execute([value])
            self.assertEqual(result,75);self.assertEqual(self.observations(),[])
    def test_input_identity_mismatch_stops_before_request(self):
        self.execute([body(H1)],hashes=[H1])
        with self.assertRaises(ValueError):self.execute([],hashes=[H2])
    def test_credential_not_written_to_database_or_output(self):
        self.execute([body(H1)],hashes=[H1])
        for f in self.root.rglob('*'):
            if f.is_file():self.assertNotIn(KEY.encode(),f.read_bytes())
    def test_redirect_is_not_followed(self):
        self.assertIsNone(vt.NoRedirect().redirect_request(None,None,302,'redirect',{},'https://example.invalid'))


if __name__=='__main__':unittest.main()
