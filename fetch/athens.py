#!/usr/bin/env python3
"""PSG-Audio (Sismanoglio hospital, Athens; Science Data Bank, CC BY 4.0) into $DIPDEPTH_DATA/psga/<patient>/ without the audio: the
technician's annotation (APNEA_RML/<patient>.rml, md5 from the release's listing) and, from each hourly EDF, only the chest and abdomen
effort and SpO2 of every 1 s record, asked for as multipart byte ranges, BATCH records a request (the audio fills 97% of each record).
Per patient: <patient>.rml, tho.i16, abd.i16, spo2.i16 (little-endian EDF digital values, the hourly files joined in order) and edf.json
(labels, gains, rates, each file's start and records). Standard library only. Resumable: a patient with edf.json is done.
usage: python3 fetch/athens.py [--workers N] [PATIENT ...]   (no PATIENT: every patient with an EDF folder)"""
import argparse, hashlib, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
API = "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"; DL = "https://download.scidb.cn/download?fileId="
DSET, VER = "778740145531650048", "V3"; WANT = ("Effort THO", "Effort ABD", "SpO2"); OUT = ["tho", "abd", "spo2"]
BATCH = 300   # 1000 ranges a request return HTTP 400
D = os.path.join(os.environ.get("DIPDEPTH_DATA", "data"), "psga")


def http(url, data=None, headers={}, tries=6):
    for k in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=120); return r.read(), r.headers
        except Exception:
            if k == tries - 1: raise
            time.sleep(5 * 2 ** k)


def ls(path):
    out, last = [], 0
    while True:
        b, _ = http(API, json.dumps(dict(dataSetId=DSET, version=VER, path=path, lastIndex=last, pageSize=500)).encode(), {"Content-Type": "application/json"})
        d = json.loads(b)["data"]; items = d["data"] if isinstance(d, dict) else d
        out += items
        if len(items) < 500: return {x["fileName"]: x for x in out}   # the listing repeats some files; duplicate names collapse to one
        last = items[-1]["index"] + 1


def header(fid):
    b, _ = http(DL + fid, headers={"Range": "bytes=0-255"}); hb = int(b[184:192]); b, _ = http(DL + fid, headers={"Range": f"bytes=0-{hb - 1}"}); return parse(b)


def parse(b):
    hb = int(b[184:192]); ns = int(b[252:256]); o = 256
    def f(w):
        nonlocal o; v = [b[o + i * w:o + (i + 1) * w].decode("latin1").strip() for i in range(ns)]; o += w * ns; return v
    L = f(16); f(80); U = f(8); pmin, pmax, dmin, dmax = (list(map(float, f(8))) for _ in range(4)); f(80); N = list(map(int, f(8)))
    off = [2 * sum(N[:i]) for i in range(ns)]
    return dict(hb=hb, nrec=int(b[236:244]), dur=float(b[244:252]), start=b[168:184].decode(), rec_bytes=2 * sum(N), labels=L, units=U, pmin=pmin, pmax=pmax, dmin=dmin, dmax=dmax, nsamp=N, off=off)


def ranges(url, at, n):
    """the n bytes at each offset of at, from one multipart range request"""
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=" + ",".join(f"{s}-{s + n - 1}" for s in at)}), timeout=300); b = r.read()
    if r.status != 206: raise IOError(f"status {r.status}")
    if not r.headers.get("Content-Type", "").startswith("multipart/byteranges"):   # one range comes back bare
        if len(at) != 1 or not r.headers.get("Content-Range", "").startswith(f"bytes {at[0]}-") or len(b) != n: raise IOError("bare response is not the range asked")
        return [b]
    got, i, lb = {}, 0, b.lower()
    while (j := lb.find(b"content-range: bytes ", i)) >= 0:
        e = b.index(b"\r\n", j); s, t = map(int, b[j + 21:e].split(b"/")[0].split(b"-")); d = b.index(b"\r\n\r\n", e) + 4
        got[s] = b[d:d + t - s + 1]; i = d + t - s + 1
    parts = [got.get(s) for s in at]
    if any(x is None or len(x) != n for x in parts): raise IOError("multipart response lacks a range")
    return parts


def fetch_file(fid, h, fh):
    """asks for the bytes from the first to the last wanted channel of BATCH records at a time and keeps the wanted channels; a failed request is retried"""
    idx = [h["labels"].index(w) for w in WANT]; a = min(h["off"][i] for i in idx); z = max(h["off"][i] + 2 * h["nsamp"][i] for i in idx)
    for k0 in range(0, h["nrec"], BATCH):
        at = [h["hb"] + k * h["rec_bytes"] + a for k in range(k0, min(k0 + BATCH, h["nrec"]))]
        for tries in range(9):
            try: parts = ranges(DL + fid, at, z - a); break
            except Exception:
                if tries == 8: raise
                time.sleep(5 * (tries + 1))
        for x in parts:
            for i, o in zip(idx, OUT): fh[o].write(x[h["off"][i] - a:h["off"][i] - a + 2 * h["nsamp"][i]])


def patient(p, rml, edfs):
    P = os.path.join(D, p); os.makedirs(P, exist_ok=True)
    if os.path.exists(os.path.join(P, "edf.json")): return p, "done"
    f = os.path.join(P, p + ".rml")
    if not os.path.exists(f):
        b, _ = http(DL + rml["id"])
        if hashlib.md5(b).hexdigest() != rml["md5"].lower(): raise IOError("annotation md5 differs from the listing")
        open(f + ".tmp", "wb").write(b); os.replace(f + ".tmp", f)
    files = sorted(edfs); hs = [header(edfs[n]["id"]) for n in files]
    if not hs or not all(h["nrec"] > 0 for h in hs): raise IOError(f"incomplete EDF listing: {len(files)} files, records {[h['nrec'] for h in hs]}")
    fh = {o: open(os.path.join(P, o + ".i16"), "wb") for o in OUT}
    for n, h in zip(files, hs): fetch_file(edfs[n]["id"], h, fh)
    for x in fh.values(): x.close()
    json.dump(dict(files=files, headers=hs), open(os.path.join(P, "edf.json.tmp"), "w")); os.replace(os.path.join(P, "edf.json.tmp"), os.path.join(P, "edf.json"))
    return p, f"{len(files)} files, {sum(h['nrec'] for h in hs)} s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PSG-Audio annotations, SpO2 and belts, without the audio")
    ap.add_argument("patients", nargs="*"); ap.add_argument("--workers", type=int, default=2); args = ap.parse_args()
    os.makedirs(D, exist_ok=True); rmls = ls(f"/{VER}/APNEA_RML"); edf_dirs = ls(f"/{VER}/APNEA_EDF")
    todo = [p for p in (args.patients or sorted(edf_dirs)) if p in edf_dirs and p + ".rml" in rmls]
    missing = sorted(set(args.patients) - set(todo))
    if not todo: raise SystemExit("nothing to fetch: the release lists no patient with both an EDF folder and an annotation")
    if missing: print("no EDF folder or annotation:", " ".join(missing), flush=True)
    def job(p):
        try: return patient(p, rmls[p + ".rml"], {n: x for n, x in ls(f"/{VER}/APNEA_EDF/{p}").items() if n.endswith(".edf")})
        except Exception as e: return p, f"FAILED {type(e).__name__}: {str(e)[:200]}"
    failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for p, msg in ex.map(job, todo): print(p, msg, flush=True); failed += msg.startswith("FAILED")
    raise SystemExit(1 if failed or missing else 0)
