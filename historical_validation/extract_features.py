#!/usr/bin/env python3
"""Trich dac trung EMBER feature version 3 (2.568 chieu) cho hai kho candidate.

Dung CHINH extractor cua tac gia (`thrember.features.PEFeatureExtractor`), khong sua mot
dong nao. Ban ghi JSONL mang `sha256` + 12 nhom dac trung tho giong het dinh dang tac gia,
cong `md5`/`sha1`/`tlsh`/`file_type`/`corpus`.

KHONG sinh `label`, `week_id`, `first_submission_date`, `detection_ratio`, `family`, tag:
tat ca deu den tu VirusTotal o T0, chua co. KHONG sinh `caps`/`ttps`/`mbc` (capa, ngoai
pham vi).

Moi truong BAT BUOC: numpy<2 va signify==0.7.1. numpy>=2 (NEP 50) lam lech
`strings.entropy` duoi 1 ULP -> khac chieu 618; signify>=0.8 vo ngay luc import.

Subcommands: selftest | parity | plan | run | verify
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

OPS = Path(r"E:\phase3\OPS_EMBER_FEATURES_20260915")
OUT = OPS / "out"
THREMBER = Path(r"E:\ember2024\EMBER2024-main\EMBER2024-main\src\thrember\features.py")
INDEX = Path(r"E:\phase3\OPS_DROP_INELIGIBLE_20260915\PE_CORPUS_PATH_INDEX_R4.parquet")
MANIFEST = Path(r"E:\phase3\STUDY_B_VT_ENGINE_LONGITUDINAL\T0_MASTER_BUILD_R1_20260914"
                r"\STAGE1_hash_manifest_R1.csv")
RELEASED_JSONL = Path(r"E:\project_data\jsonl_clean-week")
EMBER_BIN = Path(r"E:\project_data\PE\EMBER2024")

CANDIDATE_CORPORA = ("mb_malware_candidate", "benign_reference_candidate")
GROUPS = ("histogram", "byteentropy", "strings", "general", "header", "section",
          "imports", "exports", "datadirectories", "richheader", "authenticode",
          "pefilewarnings")
SHA_RE = re.compile(rb'"sha256"\s*:\s*"([0-9a-f]{64})"')
N_SHARDS = 256

_EX = None


def extractor():
    """Nap features.py cua tac gia mot lan cho moi tien trinh."""
    global _EX
    if _EX is None:
        spec = importlib.util.spec_from_file_location("thrember_features", THREMBER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EX = mod.PEFeatureExtractor()
    return _EX


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def enforce_hashseed() -> None:
    """Ghim PYTHONHASHSEED=0 bang cach tu khoi dong lai neu chua co.

    BAT BUOC, khong phai toi uu: `PEFormatWarnings` giu danh muc trong `set()` va duyet
    `for suf in self.warning_suffixes: ... break`.  Danh muc co muc CHONG LAN
    ('symbol entries. Assuming corrupt.' ket thuc bang 'Assuming corrupt.'), nen muc nao
    thang phu thuoc THU TU DUYET SET, tuc phu thuoc hash randomization - ngau nhien moi
    tien trinh.  Da chung minh: cung mot tep cho vector[2564:2568] = [0,0,1,3] hay
    [1,0,0,3] tuy PYTHONHASHSEED.  Khong ghim thi bo dac trung KHONG tai lap duoc.

    Phai dat TRUOC khi trinh thong dich khoi dong, nen `os.environ[...]` giua chung
    khong co tac dung -> phai chay lai tien trinh.

    Dung subprocess chu KHONG dung os.execv: tren Windows execv THAY THE tien trinh, nen
    shell thay lenh ket thuc ngay, mat chuyen huong stdout, va tien trinh con chay tach
    roi khong quan sat duoc.  subprocess giu tien trinh cha lam vo boc: stdout/stderr noi
    thang, ma tra ve duoc truyen lai.
    """
    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    import subprocess
    env = dict(os.environ, PYTHONHASHSEED="0")
    print("[env] dat PYTHONHASHSEED=0 va chay lai (bat buoc cho tinh tat dinh)", flush=True)
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)


def check_env() -> dict:
    import importlib.metadata as md

    import numpy as np
    import pefile
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pefile": pefile.__version__, "signify": md.version("signify"),
           "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}
    if env["PYTHONHASHSEED"] != "0":
        raise SystemExit("DUNG: PYTHONHASHSEED=%r, can '0'. Thieu no thi nhom "
                         "pefilewarnings bat dinh giua cac tien trinh."
                         % env["PYTHONHASHSEED"])
    if int(np.__version__.split(".")[0]) >= 2:
        raise SystemExit("DUNG: numpy %s >= 2. Can numpy<2, neu khong `strings.entropy` "
                         "lech va chieu 618 khac voi EMBER2024." % np.__version__)
    if env["signify"] != "0.7.1":
        raise SystemExit("DUNG: signify %s, can dung 0.7.1." % env["signify"])
    return env


def record_for(path: str, sha_expect: str, corpus: str, file_type: str) -> dict:
    """Doc file MOT lan; tra ve ban ghi JSONL hoac dict loi."""
    with open(path, "rb") as fh:
        b = fh.read()
    raw = extractor().raw_features(b)
    if raw["sha256"] != sha_expect:
        return {"_error": "SHA_MISMATCH", "path": path, "expected": sha_expect,
                "got": raw["sha256"]}
    import tlsh as _tlsh
    try:
        t = _tlsh.hash(b)
    except Exception:                                              # noqa: BLE001
        t = None
    rec = {"md5": hashlib.md5(b).hexdigest(), "sha1": hashlib.sha1(b).hexdigest(),
           "sha256": raw["sha256"], "tlsh": (t or None), "file_type": file_type,
           "corpus": corpus}
    for g in GROUPS:
        rec[g] = raw[g]
    return rec


# --------------------------------------------------------------------- work list
def load_worklist() -> list:
    import pyarrow.parquet as pq
    t = pq.read_table(INDEX, columns=["sha256", "corpus", "path"])
    sha = t.column("sha256").to_pylist()
    cor = t.column("corpus").to_pylist()
    pat = t.column("path").to_pylist()
    ft = {}
    with open(MANIFEST, encoding="utf8", newline="") as fh:
        import csv
        for r in csv.DictReader(fh):
            if r["corpus"] in CANDIDATE_CORPORA:
                ft[r["sha256"]] = r["file_type_derived"]
    work = [(s, c, p, ft.get(s, "UNKNOWN"))
            for s, c, p in zip(sha, cor, pat) if c in CANDIDATE_CORPORA]
    work.sort(key=lambda r: r[0])
    return work


def shard_of(sha: str) -> int:
    return int(sha[:4], 16) % N_SHARDS


def do_shard(args) -> dict:
    idx, items = args
    check_env()
    tmp = OUT / ("shard_%04d.jsonl.tmp" % idx)
    final = OUT / ("shard_%04d.jsonl" % idx)
    n = 0
    errs = []
    t0 = time.perf_counter()
    nbytes = 0
    with open(tmp, "w", encoding="utf8", newline="\n") as fh:
        for sha, corpus, path, ft in items:
            try:
                nbytes += os.path.getsize(path)
                rec = record_for(path, sha, corpus, ft)
            except Exception as ex:                                # noqa: BLE001
                errs.append({"sha256": sha, "error": repr(ex)[:200]})
                continue
            if "_error" in rec:
                errs.append(rec)
                continue
            fh.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":")) + "\n")
            n += 1
    os.replace(tmp, final)
    (OUT / ("shard_%04d.done" % idx)).write_text(
        json.dumps({"records": n, "errors": errs, "bytes_read": nbytes,
                    "elapsed_sec": round(time.perf_counter() - t0, 1)}), encoding="utf8")
    return {"shard": idx, "records": n, "errors": len(errs), "bytes": nbytes,
            "sec": time.perf_counter() - t0}


# ---------------------------------------------------------------------- run/plan
def cmd_plan(args) -> int:
    work = load_worklist()
    by_corpus = {}
    for _, c, _, _ in work:
        by_corpus[c] = by_corpus.get(c, 0) + 1
    tot = sum(os.path.getsize(p) for _, _, p, _ in work[:2000])
    rep = {"record_type": "EXTRACT_PLAN", "generated_utc": utcnow(),
           "files": len(work), "by_corpus": by_corpus, "shards": N_SHARDS,
           "avg_size_sample_MB": round(tot / 2000 / 1e6, 3),
           "env": check_env()}
    (OPS / "PLAN.json").write_text(json.dumps(rep, indent=2), encoding="utf8")
    print(json.dumps(rep, indent=2))
    return 0


def cmd_run(args) -> int:
    env = check_env()
    OUT.mkdir(parents=True, exist_ok=True)
    work = load_worklist()
    if args.limit:
        work = work[: args.limit]
    shards = {}
    for item in work:
        shards.setdefault(shard_of(item[0]), []).append(item)
    todo = [(i, v) for i, v in sorted(shards.items())
            if not (OUT / ("shard_%04d.done" % i)).exists()]
    print("[run] env %s | %d tep, %d shard, %d shard con lai, %d worker"
          % (env, len(work), len(shards), len(todo), args.workers))
    if not todo:
        print("[run] khong con gi de lam")
        return 0
    t0 = time.perf_counter()
    done = 0
    nrec = 0
    nerr = 0
    nbytes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(do_shard, todo):
            done += 1
            nrec += r["records"]
            nerr += r["errors"]
            nbytes += r["bytes"]
            el = time.perf_counter() - t0
            if done % 8 == 0 or done == len(todo):
                print("  shard %d/%d | %d ban ghi | %d loi | %.1f GB | %.0fs | %.1f MB/s"
                      % (done, len(todo), nrec, nerr, nbytes / 1e9, el, nbytes / el / 1e6),
                      flush=True)
    out = {"record_type": "EXTRACT_RUN", "finished_utc": utcnow(), "env": env,
           "shards_done": done, "records": nrec, "errors": nerr,
           "bytes_read": nbytes, "elapsed_sec": round(time.perf_counter() - t0, 1)}
    (OPS / "RUN_SUMMARY.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    print(json.dumps(out, indent=2))
    return 0 if nerr == 0 else 2


# ---------------------------------------------------------------------- parity
def _leaf_diffs(a, b, path=""):
    """Sinh (duong_dan, gia_tri_minh, gia_tri_tacgia, la_so_thuc) tai moi la khac nhau."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                yield (path + "/" + str(k), a.get(k), b.get(k), False)
            else:
                yield from _leaf_diffs(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (path, "len %d" % len(a), "len %d" % len(b), False)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from _leaf_diffs(x, y, path + "/%d" % i)
    else:
        if a != b:
            both_float = isinstance(a, float) and isinstance(b, float)
            yield (path, a, b, both_float)


def _parity_one(args) -> dict:
    """So sanh mot mau o HAI muc: vector float32 (quyet dinh) va dac trung tho (chan doan)."""
    sha, path, released = args
    check_env()
    ex = extractor()
    with open(path, "rb") as fh:
        b = fh.read()
    raw = ex.raw_features(b)
    if raw["sha256"] != sha:
        return {"sha256": sha, "fatal": "SHA_MISMATCH"}

    # --- tieu chi QUYET DINH: vector float32, dung chinh vectorizer cua tac gia ---
    import numpy as np
    v_ours = ex.process_raw_features(raw)
    v_theirs = ex.process_raw_features(released)
    ne = np.where(v_ours != v_theirs)[0]

    # --- chan doan: dac trung tho, phan loai la khac nhau ---
    raw_groups, float_leaves, other_leaves, max_rel = [], 0, [], 0.0
    for g in GROUPS:
        if json.dumps(raw[g], sort_keys=True) == json.dumps(released[g], sort_keys=True):
            continue
        raw_groups.append(g)
        for p, mine, theirs, is_float in _leaf_diffs(raw[g], released[g], g):
            if is_float:
                float_leaves += 1
                denom = max(abs(mine), abs(theirs), 1e-300)
                max_rel = max(max_rel, abs(mine - theirs) / denom)
            else:
                other_leaves.append({"path": p, "mine": repr(mine)[:60],
                                     "theirs": repr(theirs)[:60]})
    return {"sha256": sha, "vec_diff_dims": ne.tolist()[:8], "vec_diff_n": int(len(ne)),
            "raw_groups": raw_groups, "float_leaf_diffs": float_leaves,
            "other_leaf_diffs": other_leaves[:5], "other_leaf_n": len(other_leaves),
            "max_rel_err": max_rel}


def cmd_parity(args) -> int:
    """Cong kiem dinh: trich lai tu nhi phan EMBER2024 thu hoi va so voi JSONL tac gia."""
    check_env()
    ondisk = {}
    for sub in ("train", "test", "challenge"):
        d = EMBER_BIN / sub
        if d.exists():
            with os.scandir(d) as it:
                for e in it:
                    if e.is_file(follow_symlinks=False):
                        ondisk[e.name] = e.path
    print("[parity] nhi phan EMBER2024 tren dia: %d" % len(ondisk))

    files = sorted(RELEASED_JSONL.glob("*.jsonl"))
    if args.shuffle_seed:
        # Ten tep sap theo NGAY, nen quet tuan tu roi dung som se chi lay tuan dau.
        # Xao tron co hat giong -> mau trai deu 64 tuan va ca 6 dinh dang.
        import random as _r
        _r.Random(args.shuffle_seed).shuffle(files)
    print("[parity] tep JSONL phat hanh: %d (shuffle_seed=%s)" % (len(files), args.shuffle_seed))

    # Xu ly THEO TUNG TEP JSONL: gom cap cua rieng tep do, so sanh, roi giai phong.
    # Gom het 153k cap vao RAM truoc se ton hang chuc GB va mot dot pickle khong lo.
    t0 = time.perf_counter()
    npairs = raw_exact = float_only = total_float_leaves = 0
    vec_fail, struct_fail, fatal = [], [], []
    max_rel = 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, f in enumerate(files):
            batch = []
            with open(f, "rb") as fh:
                for line in fh:
                    m = SHA_RE.search(line[:400])
                    if not m:
                        continue
                    sha = m.group(1).decode()
                    if sha in ondisk:
                        rec = json.loads(line)
                        batch.append((sha, ondisk[sha], {g: rec[g] for g in GROUPS}))
            if batch:
                for r in pool.map(_parity_one, batch, chunksize=8):
                    npairs += 1
                    if r.get("fatal"):
                        fatal.append(r)
                        continue
                    if not r["raw_groups"]:
                        raw_exact += 1
                    if r["vec_diff_n"]:
                        vec_fail.append(r)
                    if r["other_leaf_n"]:
                        struct_fail.append(r)
                    elif r["raw_groups"]:
                        float_only += 1
                    total_float_leaves += r["float_leaf_diffs"]
                    max_rel = max(max_rel, r["max_rel_err"])
            del batch
            if (k + 1) % 20 == 0 or k == len(files) - 1:
                print("  tep %d/%d | %d cap | raw khop %d | VECTOR lech %d | CAU TRUC lech %d | %.0fs"
                      % (k + 1, len(files), npairs, raw_exact, len(vec_fail),
                         len(struct_fail), time.perf_counter() - t0), flush=True)
            if args.limit and npairs >= args.limit:
                break
    rep = {"record_type": "EXTRACT_PARITY", "verified_utc": utcnow(), "env": check_env(),
           "criterion": ("PASS = 0 chieu vector float32 khac VA 0 la khong-phai-so-thuc khac. "
                         "Lech chu so cuoi cua float64 (entropy) duoc BAO CAO chu khong fail: "
                         "no do libm cua he dieu hanh, khong do dac trung."),
           "pairs": npairs,
           "raw_bit_exact": raw_exact,
           "vector_mismatched": len(vec_fail),
           "structural_mismatched": len(struct_fail),
           "float_only_mismatched": float_only,
           "total_float_leaf_diffs": total_float_leaves,
           "max_relative_error": max_rel,
           "fatal_n": len(fatal),
           "vector_examples": vec_fail[:10],
           "structural_examples": struct_fail[:10],
           "elapsed_sec": round(time.perf_counter() - t0, 1)}
    rep["VERDICT"] = ("PASS" if npairs > 0 and not vec_fail and not struct_fail and not fatal
                      else "FAIL")
    (OPS / "PARITY_REPORT.json").write_text(json.dumps(rep, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("vector_examples", "structural_examples")}, indent=2))
    return 0 if rep["VERDICT"] == "PASS" else 2


# ---------------------------------------------------------------------- verify
def cmd_verify(args) -> int:
    import random
    work = {s: (c, p, f) for s, c, p, f in load_worklist()}
    seen = set()
    dup = 0
    nrec = 0
    for f in sorted(OUT.glob("shard_*.jsonl")):
        with open(f, "rb") as fh:
            for line in fh:
                m = SHA_RE.search(line[:400])
                sha = m.group(1).decode()
                nrec += 1
                if sha in seen:
                    dup += 1
                seen.add(sha)
    missing = set(work) - seen
    extra = seen - set(work)

    rng = random.Random(args.seed)
    sample = rng.sample(sorted(seen), min(args.sample, len(seen)))
    idx = {}
    for f in sorted(OUT.glob("shard_*.jsonl")):
        with open(f, "rb") as fh:
            for line in fh:
                m = SHA_RE.search(line[:400])
                if m and m.group(1).decode() in set(sample):
                    idx[m.group(1).decode()] = json.loads(line)
    bad = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        args_it = [(s, work[s][1], {g: idx[s][g] for g in GROUPS}) for s in sample if s in idx]
        for r in pool.map(_parity_one, args_it, chunksize=8):
            # cung tieu chi nhu cong parity: vector float32 + la khong-phai-so-thuc
            if r.get("fatal") or r["vec_diff_n"] or r["other_leaf_n"]:
                bad.append(r)

    checks = {
        "records_equals_worklist": (nrec == len(work), nrec, len(work)),
        "no_duplicate_sha256": (dup == 0, dup, 0),
        "none_missing": (not missing, len(missing), 0),
        "none_extra": (not extra, len(extra), 0),
        "resample_reproduces": (not bad, len(bad), 0),
    }
    rep = {"record_type": "EXTRACT_VERIFY", "verified_utc": utcnow(),
           "records": nrec, "unique_sha256": len(seen), "resample_size": len(sample),
           "checks": {k: {"pass": bool(a), "got": b, "want": c} for k, (a, b, c) in checks.items()}}
    rep["VERDICT"] = "PASS" if all(a for a, _, _ in checks.values()) else "FAIL"
    (OPS / "VERIFY_REPORT.json").write_text(json.dumps(rep, indent=2), encoding="utf8")
    print(json.dumps(rep, indent=2))
    return 0 if rep["VERDICT"] == "PASS" else 2


# -------------------------------------------------------------------- selftest
def cmd_selftest(args) -> int:
    env = check_env()
    ex = extractor()
    checks = [("moi truong numpy<2", not env["numpy"].startswith("2")),
              ("signify == 0.7.1", env["signify"] == "0.7.1"),
              ("pefile == 2024.8.26", env["pefile"] == "2024.8.26"),
              ("extractor 2568 chieu", ex.dim == 2568)]
    d = EMBER_BIN / "train"
    one = None
    with os.scandir(d) as it:
        for e in it:
            if e.is_file(follow_symlinks=False):
                one = e
                break
    if one:
        with open(one.path, "rb") as fh:
            b = fh.read()
        raw = ex.raw_features(b)
        v = ex.process_raw_features(raw)
        checks += [("sha256 tu extractor khop ten tep", raw["sha256"] == one.name),
                   ("du 12 nhom", all(g in raw for g in GROUPS)),
                   ("vector dai 2568", len(v) == 2568)]
    ok = all(c[1] for c in checks)
    for n, p in checks:
        print("  [%s] %s" % ("PASS" if p else "FAIL", n))
    print("SELFTEST:", "PASS" if ok else "FAIL", "|", env)
    return 0 if ok else 1


def main() -> int:
    enforce_hashseed()          # tu exec lai neu chua co PYTHONHASHSEED=0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("plan"); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("parity"); p.set_defaults(fn=cmd_parity)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=20260915,
                   help="xao tron thu tu tep JSONL de --limit lay mau trai deu; 0 = giu nguyen")
    p = sub.add_parser("run"); p.set_defaults(fn=cmd_run)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("verify"); p.set_defaults(fn=cmd_verify)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()
    OPS.mkdir(parents=True, exist_ok=True)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
