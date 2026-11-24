#!/usr/bin/env bash
# APSAA, Dr. Sagaz Hospital, Jaén (Zenodo record 14096541, CC BY 4.0): its one zip (3.9 GB, audio included) into $DIPDEPTH_DATA/apsaa/, md5 from the record. Resumable.
# usage: bash fetch/jaen.sh
set -u
D="${DIPDEPTH_DATA:-data}/apsaa"; mkdir -p "$D" && cd "$D" || exit 1
URL=https://zenodo.org/api/records/14096541/files/APSAA.zip/content
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [ -s APSAA.zip ] && break
  # Zenodo can drop a long transfer, so each retry resumes from the partial file
  curl -sSfL -C - -o APSAA.zip.part "$URL" && mv APSAA.zip.part APSAA.zip || sleep 10
done
[ -s APSAA.zip ] || { echo "failed: $URL"; exit 1; }
echo "44420c64647f6b7d898d3e84ff48e910  APSAA.zip" | md5sum --quiet -c -
