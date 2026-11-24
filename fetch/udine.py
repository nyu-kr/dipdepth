#!/usr/bin/env python3
"""OSASUD, Udine University Hospital stroke unit (figshare collection 5630890, CC0): its one pandas pickle (3.3 GB) into
$DIPDEPTH_DATA/osasud/, md5 from figshare, then one npz per patient in osasud/npz/ on a gapless one-second grid: spo2 (1 Hz),
thorax and abdomen (10 Hz), event (a label per second) and start. The unpickler admits exactly the 15 globals this file needs and
refuses any other. Resumable. The pickle was written by pandas 1.x and pandas 2 cannot load it, so this runs in its own environment
with pandas 1.5 (tested: Python 3.10, pandas 1.5.3, numpy 1.23.5):
    python3.10 -m venv osasud-env && osasud-env/bin/pip install pandas==1.5.3 numpy==1.23.5
usage: osasud-env/bin/python fetch/udine.py"""
import collections, hashlib, os, pickle, sys, urllib.request
import numpy as np
URL, MD5 = "https://ndownloader.figshare.com/files/34135113", "b9ed560f4abf6100ce02cfc7cb0c121f"
D = os.path.join(os.environ.get("DIPDEPTH_DATA", "data"), "osasud"); PICKLE = os.path.join(D, "dataset_OSAS.pickle"); NPZ = os.path.join(D, "npz")


class Restricted(pickle.Unpickler):
    """admits only the globals the published pickle needs; any other refuses the file"""
    ALLOWED = {("builtins", "slice"), ("functools", "partial"), ("numpy", "dtype"), ("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct"),
               ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"), ("pandas.core.arrays.boolean", "BooleanArray"),
               ("pandas.core.arrays.boolean", "BooleanDtype"), ("pandas.core.arrays.datetimes", "DatetimeArray"), ("pandas.core.frame", "DataFrame"),
               ("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.numeric", "Int64Index"),
               ("pandas.core.internals.blocks", "new_block"), ("pandas.core.internals.managers", "BlockManager")}
    def find_class(self, module, name):
        if (module, name) in self.ALLOWED: return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def convert():
    """the pickle through Restricted into one npz per patient"""
    with open(PICKLE, "rb") as f: df = Restricted(f).load()
    os.makedirs(NPZ, exist_ok=True); counts = collections.Counter(); info = {}
    for pid, g in df.groupby("patient", sort=False):
        t = ((g["timestamp_datetime"] - g["timestamp_datetime"].min()).dt.total_seconds()).to_numpy().round().astype(int); assert len(set(t)) == len(t), f"{pid}: duplicate seconds"
        n = int(t.max()) + 1; spo2 = np.full(n, np.nan); spo2[t] = g["SpO2(%)"].to_numpy(float); event = np.full(n, "NONE", dtype="U40"); event[t] = g["event"].astype(str).to_numpy()
        belts = {}
        for col, nm in (("PSG_Thorax", "thorax"), ("PSG_Abdomen", "abdomen")):
            v = np.full((n, 10), np.nan)
            for i, a in zip(t, g[col]):
                if a is not None and np.shape(a) == (10,): v[i] = a
            belts[nm] = v.ravel()
        np.savez(os.path.join(NPZ, f"{pid}.npz"), spo2=spo2, event=event, **belts, start=str(g["timestamp_datetime"].min()))
        counts.update(g["event"].astype(str)); info[pid] = (n, len(t))
    print(len(info), "patients;", sum(n != m for n, m in info.values()), "with gaps in their seconds;", dict(counts))


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    if not os.path.exists(PICKLE):
        urllib.request.urlretrieve(URL, PICKLE + ".part"); os.replace(PICKLE + ".part", PICKLE)
    if md5(PICKLE) != MD5: sys.exit(f"md5 of {PICKLE} differs from figshare's {MD5}")
    convert()
