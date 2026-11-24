# DipDepth

Code and per-night results for the paper *DipDepth: Sleep Apnea Beyond the 3% Rule*.

DipDepth reads an overnight finger oxygen saturation (SpO2) trace. Sparse deconvolution recovers oxygen-loss pulses through a drop shape fitted to each night, a climb-back rule counts events, and a marked point process gives the expected drop in every 10 s bin. The paper compares it with the 3% rule on six public sleep datasets and describes how each test was registered and timestamped before scoring; the registrations, fingerprints and RFC 3161 timestamp receipts are not part of this repository and are available from the author on request.

## Contents

- `dipdepth/model.py`: the model as the paper ran it (step 1, the event count, step 2 and the 3% rule).
- `dipdepth/readers.py`: SpO2 at 1 Hz, scored respiratory events, sleep stages and belt variability for the six datasets.
- `dipdepth/measures.py`: the per-night measures and the paired bootstrap statistics.
- `dipdepth/readings.py`: the ablation's readings (pulse depths, area below the level, the rule's summed dips).
- `fetch/`: one download script per dataset.
- `run_night.py`: DipDepth on one night, from SpO2 alone.
- `score_night.py`: one night's row of `results/`, scored against its events, stages and belts.
- `results/`: one row per night for each test, the fitted parameters of the sealed nights, the split, and the values the tables must reproduce (`results/README.md` lists every column).
- `tables.py`: rebuilds the paper's Tables II to IV from `results/`, with the text's figures drawn from the same rows (nights higher and lower, the gain by hospital, the ablation, the expected drop's per-window and screening comparisons, the duplicate night), and exits nonzero if any value differs from `results/expected.json`. The split's totals, the simulated nights and the check-night figures are not rebuilt.
- `fig_night.py` and `examples/tr03-0092.npz`: Fig. 1, from the arrays of one Boston night.
- `tests/`: the model and the 3% rule on the example night, and the tables, without downloads.

## Run

Python 3.10 or later:

    pip install -r requirements.txt
    python3 tables.py
    python3 fig_night.py
    python3 -m pytest tests

One night from the raw data:

    export DIPDEPTH_DATA=data
    bash fetch/boston.sh tr03-0092
    python3 run_night.py c2018:tr03-0092
    python3 score_night.py c2018:tr03-0092

Keys are `DATASET:RECORD`: `c2018:` Boston, `psga:` Athens, `ucd:` Dublin, `apsaa:` Jaén, `isleeps:` Bengaluru, `osasud:` Udine. Each script in `fetch/` gives its usage in its first lines. Downloads are checked against the checksums each source publishes; at Athens these cover the annotations only, since its signals are read in byte ranges. Running a script again skips completed files, so delete a file that failed its check first. The Udine pickle needs pandas 1.5 for its one-time conversion (`fetch/udine.py` says how).

A night takes about 17 CPU-seconds. The paper ran Python 3.13.5 with numpy 2.1.3, scipy 1.15.3, pandas 2.2.3, wfdb 4.3.1 and openpyxl 3.1.5; with these versions on one thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`) the model reproduces the paper's stored predictions bit for bit. With numpy 2.5 and scipy 1.18 the example night's run depths and expected drop move by under 1e-6; the tests allow 1e-5.

## Data

No recording is bundled except the example night. Each dataset keeps its own licence; cite it as it asks:

| Hospital | Dataset (release used) | Licence | Cite |
|---|---|---|---|
| Boston | PhysioNet/CinC Challenge 2018, version 1.0.0, doi:10.13026/6phb-r450 | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) | Ghassemi et al., Computing in Cardiology 45 (2018), doi:10.22489/CinC.2018.049; for PhysioNet, Pollard et al., Nature Health 1:792 (2026), doi:10.1038/s44360-026-00096-z |
| Athens | PSG-Audio, Science Data Bank V3, doi:10.57760/sciencedb.00345 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Korompili et al., Scientific Data 8:197 (2021), doi:10.1038/s41597-021-00977-w |
| Dublin | St Vincent's University Hospital / UCD Sleep Apnea Database, version 1.0.0, doi:10.13026/C26C7D | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) | McNicholas et al., PhysioNet (2004), doi:10.13026/C26C7D; for PhysioNet, Pollard et al. (2026) |
| Jaén | APSAA, Zenodo version 2.0, doi:10.5281/zenodo.14096541 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and the record's conditions: academic, non-commercial use, no redistribution of the database | Gonzalez-Martinez et al., EURASIP J. Audio Speech Music Process. 2025:24, doi:10.1186/s13636-025-00410-9 |
| Bengaluru | iSLEEPS, doi:10.5281/zenodo.14873844 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Maiti et al., Scientific Data 13:421 (2026), doi:10.1038/s41597-026-06747-w |
| Udine | OSASUD, doi:10.6084/m9.figshare.19214289.v1 | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | Bernardini et al., Scientific Data 9:177 (2022), doi:10.1038/s41597-022-01272-y |

`examples/tr03-0092.npz` holds record tr03-0092 of the PhysioNet/CinC Challenge 2018 (version 1.0.0, doi:10.13026/6phb-r450): its SpO2 at 1 Hz (each second's median), its 181 technician-scored respiratory events, and the arrays DipDepth and the 3% rule computed from them; Fig. 1 draws them. It is shared under the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/), with the Boston attribution above.

The files in `results/` hold per-night statistics computed from the six datasets (AHI, event counts, F1 scores, rank correlations, fitted parameters), no signal or annotation. They are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with the attributions above, except the 32 Jaén rows, which follow the APSAA record's conditions: academic, non-commercial use.

## Licence and citation

Code is MIT (`LICENSE`). Cite the paper; `CITATION.cff` gives the entry.
