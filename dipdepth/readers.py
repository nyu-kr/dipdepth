"""One night's recordings from the six datasets: SpO2 at 1 Hz, the scored respiratory events and sleep stages, and the breathing belts.
A night's key is DATASET:RECORD, DATASET one of c2018 (Boston), psga (Athens), ucd (Dublin), apsaa (Jaén), isleeps (Bengaluru) and
osasud (Udine). Files are read from $DIPDEPTH_DATA (default ./data), one folder per dataset as the fetch/ scripts lay them out."""
import datetime, functools, io, json, math, os, re, warnings, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("DIPDEPTH_DATA", "data"))
WINDOW = 600
# respiratory event rows of Jaén, Bengaluru and Udine; every other label (RERA, snore, desaturation, "No Effort", body event, blank, NONE) is dropped
KEEP = {"apsaa": {"Hypopnea", "Obstructive Apnea", "Central Apnea", "Mixed Apnea"}, "isleeps": {"Hypopnea", "Obstructive Apnea", "Central Apnea", "Mixed Apnea"},
        "osasud": {"HYPOPNEA", "APNEA-CENTRAL", "APNEA-OBSTRUCTIVE", "APNEA-MIXED"}}


def split_key(key):
    """(dataset, record) of a night key DATASET:RECORD"""
    site, _, rec = key.partition(":")
    if site not in ("c2018", "psga", "ucd", *KEEP) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", rec) or ".." in rec:
        raise ValueError(f"night key {key!r}: expected DATASET:RECORD, DATASET one of c2018, psga, ucd, apsaa, isleeps, osasud (e.g. c2018:tr03-0092)")
    return site, rec


# Boston, Athens and Dublin
def cinc(rec):
    """a Boston night's signal file, channel names, (gain, baseline) per channel, samples and header offset"""
    folder = ROOT / "c2018" / ("test" if rec.startswith("te") else "training") / rec
    lines = [l.split() for l in (folder / (rec + ".hea")).read_text().splitlines() if not l.startswith("#")]
    gains = {l[-1]: (float(l[2].split("(")[0].split("/")[0]), int(l[2].split("(")[1].split(")")[0]) if "(" in l[2] else 0) for l in lines[1:]}
    return folder / (rec + ".mat"), [l[-1] for l in lines[1:]], gains, int(lines[0][3]), int(lines[1][1].split("+")[1])


def calibration(header, label):
    assert header["labels"].count(label) == 1, "missing or duplicate EDF channel: " + label
    i = header["labels"].index(label); gain = (header["pmax"][i] - header["pmin"][i]) / (header["dmax"][i] - header["dmin"][i])
    offset = header["pmin"][i] - gain * header["dmin"][i]; rate = header["nsamp"][i] / header["dur"]; unit = header["units"][i].strip().lower()
    assert math.isfinite(gain) and gain > 0 and math.isfinite(offset) and rate > 0
    return gain, offset, rate, unit


def athens_headers(rec):
    """an Athens night's hourly EDF headers, refused when a gain, offset, rate or unit changes or the hours leave a gap"""
    headers = json.loads((ROOT / "psga" / rec / "edf.json").read_text())["headers"]
    assert headers and all(h["nrec"] > 0 and h["dur"] > 0 for h in headers)
    labels = ("Effort THO", "Effort ABD", "SpO2"); expected = {l: calibration(headers[0], l) for l in labels}
    for h in headers[1:]:
        for l in labels:
            actual = calibration(h, l)
            assert actual[3] == expected[l][3], (l, "hourly EDF unit changed")
            assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10) for a, b in zip(actual[:3], expected[l][:3])), (l, "hourly EDF gain, offset or rate changed")
    for a, b in zip(headers, headers[1:]):
        gap = datetime.datetime.strptime(b["start"], "%d.%m.%y%H.%M.%S") - datetime.datetime.strptime(a["start"], "%d.%m.%y%H.%M.%S")
        assert math.isclose(gap.total_seconds(), a["nrec"] * a["dur"], abs_tol=1.), "hourly EDF gap or overlap"
    return headers


def athens_channel(rec, name):
    h = athens_headers(rec)[0]; i = h["labels"].index({"spo2": "SpO2", "tho": "Effort THO", "abd": "Effort ABD"}[name])
    gain = (h["pmax"][i] - h["pmin"][i]) / (h["dmax"][i] - h["dmin"][i]); raw = np.fromfile(ROOT / "psga" / rec / (name + ".i16"), "<i2").astype(float)
    return (raw - h["dmin"][i]) * gain + h["pmin"][i], h["nsamp"][i] / h["dur"]


def edf_channel(path, want):
    """a Dublin EDF channel (signal, rate, start); whole records only, so a truncated last record is dropped"""
    with open(path, "rb") as f:
        h = f.read(256); start, stime = h[168:176].decode("latin1"), h[176:184].decode("latin1"); dur = float(h[244:252]); ns = int(h[252:256])
        labels = [f.read(16).decode("latin1").strip() for _ in range(ns)]; f.read(80 * ns); f.read(8 * ns)
        pmin = [float(f.read(8)) for _ in range(ns)]; pmax = [float(f.read(8)) for _ in range(ns)]
        dmin = [int(float(f.read(8))) for _ in range(ns)]; dmax = [int(float(f.read(8))) for _ in range(ns)]
        f.read(80 * ns); nsamp = [int(f.read(8)) for _ in range(ns)]; f.read(32 * ns); data = f.read()
    norm = lambda l: l.upper().replace(" ", ""); c = [i for i, l in enumerate(labels) if norm(l) in {norm(w) for w in want}][0]
    per = sum(nsamp); nr = len(data) // (2 * per); off = sum(nsamp[:c])
    raw = np.frombuffer(data, "<i2", count=nr * per).reshape(nr, per)[:, off:off + nsamp[c]].ravel().astype(float)
    gain = (pmax[c] - pmin[c]) / (dmax[c] - dmin[c]) if dmax[c] != dmin[c] else 1.0
    d, m, y = [int(v) for v in start.split(".")]; y = y + 2000 if y < 85 else y + 1900
    return (raw - dmin[c]) * gain + pmin[c], nsamp[c] / dur, datetime.datetime(y, m, d, *[int(v) for v in stime.split(".")])


def rml_references(path):
    """an Athens night's respiratory events (start, end, type) and the technician's stages from its RML file"""
    tree = ET.parse(path); local = lambda e: e.tag.rsplit("}", 1)[-1]
    users = [e for e in tree.iter() if local(e) == "UserStaging"]; assert len(users) == 1, "missing or ambiguous user staging"
    names = {"Wake": "W", "NonREM1": "N1", "NonREM2": "N2", "NonREM3": "N3", "REM": "R"}
    stages = [(float(e.attrib["Start"]), names.get(e.attrib["Type"], "U")) for e in users[0].iter() if local(e) == "Stage"]; assert stages, "no user stages decoded"
    types = {"ObstructiveApnea": "resp_obstructiveapnea", "CentralApnea": "resp_centralapnea", "MixedApnea": "resp_mixedapnea", "Hypopnea": "resp_hypopnea"}; events = []
    for e in tree.iter():
        if local(e) == "Event" and e.attrib.get("Family") == "Respiratory" and e.attrib.get("Type") in types:
            start, duration = float(e.attrib["Start"]), float(e.attrib["Duration"]); assert start >= 0 and duration > 0
            events.append((start, start + duration, types[e.attrib["Type"]]))
    return sorted(events), sorted(stages)


# Jaén, Bengaluru and Udine
def edf_header(path):
    """an EDF's header: start (date, time), reserved field, records, record seconds, and per signal label, unit, physical and digital range, samples per record"""
    with open(path, "rb") as f:
        h = f.read(256); ns = int(h[252:256]); sig = f.read(256 * ns)
    fields, o = {}, 0
    for name, w in (("labels", 16), ("transducer", 80), ("units", 8), ("pmin", 8), ("pmax", 8), ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("nsamp", 8), ("res", 32)):
        fields[name] = [sig[o + i * w: o + (i + 1) * w].decode("latin1").strip() for i in range(ns)]; o += w * ns
    return {"start": (h[168:176].decode("latin1"), h[176:184].decode("latin1")), "reserved": h[192:236].decode("latin1").strip(), "nrec": int(h[236:244]), "dur": float(h[244:252]), "ns": ns,
            "labels": fields["labels"], "units": fields["units"], "pmin": [float(v) for v in fields["pmin"]], "pmax": [float(v) for v in fields["pmax"]],
            "dmin": [int(float(v)) for v in fields["dmin"]], "dmax": [int(float(v)) for v in fields["dmax"]], "nsamp": [int(v) for v in fields["nsamp"]]}


def edf_signal(path, labels):
    """the first present of the labels as physical values and its rate (None, None when absent); whole records only"""
    h = edf_header(path); norm = lambda l: l.upper().replace(" ", "")
    i = next((h["labels"].index(l) for w in labels for l in h["labels"] if norm(l) == norm(w)), None)
    if i is None: return None, None
    per = sum(h["nsamp"]); nr = min(h["nrec"], (os.path.getsize(path) - 256 * (1 + h["ns"])) // (2 * per)); off = sum(h["nsamp"][:i])
    mm = np.memmap(path, "<i2", "r", offset=256 * (1 + h["ns"]), shape=(nr, per)); raw = np.asarray(mm[:, off:off + h["nsamp"][i]]).ravel().astype(float)
    gain = (h["pmax"][i] - h["pmin"][i]) / (h["dmax"][i] - h["dmin"][i]); return (raw - h["dmin"][i]) * gain + h["pmin"][i], h["nsamp"][i] / h["dur"]


def per_second(values, fs):
    """one value per second: the median of the second's samples, values outside 50 to 100 missing (mask, then median)"""
    fs = int(round(fs)); x = np.asarray(values, float)[:len(values) // fs * fs].reshape(-1, fs); x = np.where((x < 50) | (x > 100), np.nan, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning); return np.nanmedian(x, axis=1)


def stamp_time(v):
    from openpyxl.utils.datetime import from_excel
    return v if isinstance(v, datetime.datetime) else from_excel(v)


def isleeps_rows(name):
    """the Flow Events sheet below its column header: (start, end, event name) with Excel serials as datetimes, and the analysis start time of its header block"""
    import openpyxl
    wb = openpyxl.load_workbook(ROOT / "isleeps" / (name + ".xlsx"), read_only=True, data_only=True); rows = list(wb["Flow Events"].iter_rows(values_only=True))
    hdr = next(i for i, r in enumerate(rows) if r and r[0] == "Time start"); c = [str(v) for v in rows[hdr]].index("Event name")
    start = next(r[1] for r in rows[:hdr] if r and r[0] == "Start Time")
    return [(stamp_time(r[0]), stamp_time(r[1]), str(r[c])) for r in rows[hdr + 1:] if r and r[0] is not None and r[1] is not None], stamp_time(start)


@functools.lru_cache(maxsize=None)
def apsaa_zip():
    import zipfile; return zipfile.ZipFile(ROOT / "apsaa" / "APSAA.zip")


def apsaa_csv(member):
    import pandas as pd; return pd.read_csv(io.BytesIO(apsaa_zip().read(member)))


def apsaa_table(name, part): return apsaa_csv(f"{name}/{name}_{part}.csv")


def apsaa_rate(signal): return float(apsaa_csv("Polygraph_Sampling_Rates.csv")[signal][0])


def apsaa_rows(name):
    """Event_Name, Start_Time (hh:mm:ss from the recording's start) and Duration (s)"""
    d = apsaa_table(name, "Annotations"); h, m, sec = (d["Start_Time"].str.split(":", expand=True).astype(int)[c] for c in (0, 1, 2)); start = h * 3600 + m * 60 + sec
    return [(float(a), float(a + b), str(n)) for a, b, n in zip(start, d["Duration"], d["Event_Name"])]


def osasud_load(name):
    with np.load(ROOT / "osasud" / "npz" / (name + ".npz")) as z: return {n: z[n] for n in z.files}


def osasud_rows(name):
    """the per-second label's runs of one value, NONE excluded: (start s, end s, label)"""
    lab = osasud_load(name)["event"]; cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1; a = np.r_[0, cut]; b = np.r_[cut, len(lab)]
    return [(float(i), float(j), str(lab[i])) for i, j in zip(a, b) if lab[i] != "NONE"]


def raw_events(site, name):
    """every labelled row of a Jaén, Bengaluru or Udine night: (start s, end s, label), seconds from the first signal sample"""
    if site == "isleeps":
        rows, start = isleeps_rows(name); clock = datetime.datetime.strptime(edf_header(ROOT / "isleeps" / (name + ".edf"))["start"][1], "%H.%M.%S").time()
        t0 = min((datetime.datetime.combine(start.date() + datetime.timedelta(d), clock) for d in (-1, 0, 1)), key=lambda c: abs(c - start))   # the EDF has a clock, not a date
        return [((a - t0).total_seconds(), (b - t0).total_seconds(), n) for a, b, n in rows]
    if site == "apsaa": return apsaa_rows(name)
    return osasud_rows(name)


# the readers
def spo2(key):
    """the night's SpO2 at 1 Hz (the median of each second's samples), values outside 50 to 100 as missing
    Boston takes each second's median before discarding values outside 50 to 100, the other datasets discard first: the order the paper's runs used"""
    site, rec = split_key(key)
    if site == "c2018":
        path, names, gains, n, offset = cinc(rec); gain, baseline = gains["SaO2"]
        x = (np.memmap(path, "<i2", "r", offset=offset, shape=(n, len(names)))[:, names.index("SaO2")].astype(float) - baseline) / gain
        x = np.median(x[:len(x) // 200 * 200].reshape(-1, 200), axis=1)
    elif site == "ucd":
        x, fs, _ = edf_channel(ROOT / "ucd" / (rec + ".rec"), ["SpO2"]); fs = int(round(fs))
        x = np.where((x < 50) | (x > 100), np.nan, x); x = np.nanmedian(x[:len(x) // fs * fs].reshape(-1, fs), axis=1)
    elif site == "psga":
        x, fs = athens_channel(rec, "spo2"); assert fs == 1
    elif site == "isleeps":
        x, fs = edf_signal(ROOT / "isleeps" / (rec + ".edf"), ["SPO2"]); assert x is not None, "no SPO2 channel"; return per_second(x, fs)
    elif site == "apsaa": return per_second(apsaa_table(rec, "SpO2_EG")["Signal"].to_numpy(float), apsaa_rate("SpO2_EG"))
    else: return per_second(osasud_load(rec)["spo2"], 1)
    return np.where((x < 50) | (x > 100), np.nan, x)


def references(key):
    """the scored events, sorted (start s, end s, type), and the sleep stage changes (time s, stage), [] for apsaa, isleeps and osasud, whose stages are not read
    Boston's are every scored interval of its training annotation (respiratory types start with resp_); the others' are respiratory events only"""
    site, rec = split_key(key)
    if site == "c2018":
        import wfdb
        a = wfdb.rdann(str(ROOT / "c2018" / "training" / rec / rec), "arousal"); opened, events, stages = {}, [], []
        for sample, note in zip(a.sample, a.aux_note):
            if note.startswith("("): opened[note[1:]] = sample / 200
            elif note.endswith(")") and note[:-1] in opened: events.append((opened.pop(note[:-1]), sample / 200, note[:-1]))
            elif note in ("W", "N1", "N2", "N3", "R"): stages.append((sample / 200, note))
        return sorted(events), stages
    if site == "ucd":
        folder = ROOT / "ucd"
        with (folder / (rec + ".rec")).open("rb") as f: t0 = datetime.datetime.strptime(f.read(256)[168:184].decode("latin1"), "%d.%m.%y%H.%M.%S")
        types = {"APNEA-O": "resp_obstructiveapnea", "APNEA-C": "resp_centralapnea", "APNEA-M": "resp_mixedapnea", "HYP-O": "resp_hypopnea", "HYP-C": "resp_hypopnea", "HYP-M": "resp_hypopnea"}
        events = []
        for line in open(folder / (rec + "_respevt.txt"), errors="replace"):
            m = re.match(r"^\s*(\d\d:\d\d:\d\d)\s+(\S+)\s+(?:(PB|CS)\s+)?(\d+)\s+([\d.]+)?\s*([\d.]+)?", line.rstrip("\r\n"))
            if not m or m.group(2) not in types: continue
            t = datetime.datetime.combine(t0.date(), datetime.time(*map(int, m.group(1).split(":"))))
            if t < t0 - datetime.timedelta(hours=2): t += datetime.timedelta(days=1)
            time = (t - t0).total_seconds(); events.append((time, time + float(m.group(4)), types[m.group(2)]))
        labels = [{0: "W", 1: "R", 2: "N1", 3: "N2", 4: "N3", 5: "N3"}.get(int(l), "U") for l in (folder / (rec + "_stage.txt")).read_text().split()]
        stages = [(30 * i, l) for i, l in enumerate(labels) if i == 0 or l != labels[i - 1]]
        if labels and labels[-1] != "U": stages.append((30 * len(labels), "U"))
        return sorted(events), stages
    if site == "psga": return rml_references(ROOT / "psga" / rec / (rec + ".rml"))
    events = sorted((a, b, n) for a, b, n in raw_events(site, rec) if n in KEEP[site])
    assert all(b > a for a, b, _ in events), "event ends before it starts"; return events, []


def envelope(values, fs):
    import pandas as pd
    n = int(len(values) / fs); sampled = np.interp(np.arange(n * 10) / 10, np.arange(len(values)) / fs, values).reshape(-1, 10)
    power = ((sampled - sampled.mean(axis=1)[:, None]) ** 2).mean(axis=1)
    return np.sqrt(pd.Series(power).rolling(5, center=True, min_periods=1).mean().to_numpy())


def effort(key):
    """the belts' breathing envelopes at 1 Hz, one row per recorded belt (chest, then abdomen), rows of equal length"""
    site, rec = split_key(key)
    if site == "c2018":
        path, names, gains, n, offset = cinc(rec); data = np.memmap(path, "<i2", "r", offset=offset, shape=(n, len(names))); out = []
        for name in ("CHEST", "ABD"):
            gain, baseline = gains[name]; values = (data[:, names.index(name)].astype(float) - baseline) / gain
            out.append(envelope(values[:len(values) // 20 * 20].reshape(-1, 20).mean(axis=1), 10))
    elif site == "ucd":
        out = [envelope(*edf_channel(ROOT / "ucd" / (rec + ".rec"), [name])[:2]) for name in ("ribcage", "abdo")]
    elif site == "psga":
        out = [envelope(*athens_channel(rec, name)) for name in ("tho", "abd")]
    else:
        if site == "isleeps":
            path = ROOT / "isleeps" / (rec + ".edf"); belts = [edf_signal(path, w) for w in (["THORAX"], ["RIP ABDOMEN", "ABDOMEN"])]
        elif site == "apsaa": belts = [(apsaa_table(rec, f"{b}_EG")["Signal"].to_numpy(float), apsaa_rate(f"{b}_EG")) for b in ("Thorax", "Abdomen")]
        else: belts = [(osasud_load(rec)[c], 10.) for c in ("thorax", "abdomen")]
        belts = [b for b in belts if b[0] is not None]; assert belts, "no belt recorded"
        out = [envelope(v, fs) for v, fs in belts]
    n = min(map(len, out)); return np.asarray([x[:n] for x in out])


def belt_variability(envelopes, seconds):
    """per 10-min window the belts' mean coefficient of variation; None unless both belts are 90% valid with a positive mean (measures.effort_windows passes a lone belt twice);
    validity is counted on the envelopes, whose 5 s smoothing bridges gaps shorter than that"""
    assert np.shape(envelopes)[0] == 2 and np.shape(envelopes)[1] >= seconds, "belt extent mismatch"
    rows = []
    for start in range(0, seconds - WINDOW + 1, WINDOW):
        block = envelopes[:, start:start + WINDOW]; valid = np.isfinite(block).mean(axis=1) >= .9; means = np.nanmean(block, axis=1)
        rows.append({"start": start, "value": float(np.mean(np.nanstd(block, axis=1) / means)) if np.all(valid & (means > 0)) else None})
    return rows
