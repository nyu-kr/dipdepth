#!/usr/bin/env bash
# PhysioNet/CinC Challenge 2018 (ODC-By 1.0) into $DIPDEPTH_DATA/c2018/training/REC/ and test/REC/, every file checked against PhysioNet's SHA256SUMS.
# usage: bash fetch/boston.sh REC... | training | test   (a whole part is about 130 GB; test nights, te*, have no scored events). Resumable.
set -u
export D="${DIPDEPTH_DATA:-data}/c2018" B=https://physionet-open.s3.amazonaws.com/challenge-2018/1.0.0
[ $# -gt 0 ] || { sed -n 2,3p "$0"; exit 2; }
for part in training test; do
  mkdir -p "$D/$part" || exit 1
  for f in RECORDS SHA256SUMS; do [ -s "$D/$part/$f" ] || curl -sSf -o "$D/$part/$f" "$B/$part/$f" || { echo "failed: $B/$part/$f"; exit 1; }; done
done
recs=(); for a in "$@"; do case "$a" in training|test) mapfile -t -O "${#recs[@]}" recs < <(sed 's#/$##' "$D/$a/RECORDS");; *) recs+=("$a");; esac; done
# a record's header and signals, plus the scored annotation for a training night
want() { for r in "${recs[@]}"; do case "$r" in te*) printf 'test %s/%s.%s\n' "$r" "$r" hea "$r" "$r" mat;; *) printf 'training %s/%s.%s\n' "$r" "$r" hea "$r" "$r" mat "$r" "$r" arousal;; esac; done; }
want | xargs -P 4 -n 2 sh -c 'f="$0/$1"; [ -s "$D/$f" ] && exit 0; mkdir -p "$D/${f%/*}" && curl -sSf -o "$D/$f.part" "$B/$f" && mv "$D/$f.part" "$D/$f" || echo "failed: $B/$f"'
bad=0
for part in training test; do
  w=$(want | awk -v p="$part" '$1 == p {print $2}'); [ -n "$w" ] || continue
  sums=$(awk 'NR == FNR {w[$0]; next} ($2 in w)' <(echo "$w") "$D/$part/SHA256SUMS")
  [ "$(echo "$sums" | grep -c .)" -eq "$(echo "$w" | grep -c .)" ] || { echo "a requested $part file is not in PhysioNet's SHA256SUMS"; bad=1; }
  (cd "$D/$part" && echo "$sums" | sha256sum --quiet -c -) || bad=1
done
exit $bad
