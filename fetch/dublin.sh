#!/usr/bin/env bash
# St Vincent's University Hospital / UCD Sleep Apnea Database 1.0.0 (PhysioNet ucddb, ODC-By 1.0, 25 nights, 1.3 GB) into $DIPDEPTH_DATA/ucd/, sha256-checked. Resumable.
# usage: bash fetch/dublin.sh
set -u
D="${DIPDEPTH_DATA:-data}/ucd"; mkdir -p "$D" && cd "$D" || exit 1
export B=https://physionet-open.s3.amazonaws.com/ucddb/1.0.0
[ -s SHA256SUMS.txt ] || curl -sSf -o SHA256SUMS.txt "$B/SHA256SUMS.txt" || { echo "failed: $B/SHA256SUMS.txt"; exit 1; }
grep -v -E '\.(shtml|html)$' SHA256SUMS.txt | awk '{print $2}' | xargs -P 8 -n 1 sh -c 'f="$0"; [ -s "$f" ] || { curl -sSf -o "$f.part" "$B/$f" && mv "$f.part" "$f"; } || echo "failed: $B/$f"'
grep -v -E '\.(shtml|html)$' SHA256SUMS.txt | sha256sum --quiet -c -
