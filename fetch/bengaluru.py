#!/usr/bin/env python3
"""iSLEEPS, stroke patients, Bengaluru (Zenodo record 14873844, CC BY 4.0): each night's EDF and scored-event workbook into
$DIPDEPTH_DATA/isleeps/, every file checked against the record's md5. Standard library only. Resumable.
usage: python3 fetch/bengaluru.py [NIGHT ...]   (NIGHT as SN1; none: all 40 nights)"""
import hashlib, json, os, sys, urllib.request
RECORD = "https://zenodo.org/api/records/14873844"
D = os.path.join(os.environ.get("DIPDEPTH_DATA", "data"), "isleeps")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    files = {f["key"]: f for f in json.load(urllib.request.urlopen(RECORD, timeout=120))["files"]}
    nights = sys.argv[1:] or sorted({k.split(".")[0] for k in files if k.startswith("SN")}, key=lambda v: int(v[2:]))   # the record also holds two Excel lock files
    if not nights: sys.exit("the record lists no nights")
    bad = 0
    for name in (f"{n}.{e}" for n in nights for e in ("edf", "xlsx")):
        if name not in files: print("not in the record:", name); bad += 1; continue
        path, want = os.path.join(D, name), files[name]["checksum"].split(":")[1]
        if not (os.path.exists(path) and md5(path) == want):
            try: urllib.request.urlretrieve(files[name]["links"]["self"], path + ".part"); os.replace(path + ".part", path)
            except OSError as e: print("failed:", name, e); bad += 1; continue
        if md5(path) != want: print("md5 differs from the record:", name); bad += 1
    sys.exit(1 if bad else 0)
