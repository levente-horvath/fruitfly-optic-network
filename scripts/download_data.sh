#!/usr/bin/env bash
# Download the MaleCNS v1.0 tables needed for simulation (~1.1 GB) into data/.
set -euo pipefail

BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DIR"

for f in \
  body-annotations-male-cns-v1.0-minconf-0.5.feather \
  body-neurotransmitters-male-cns-v1.0.feather \
  connectome-weights-male-cns-v1.0-minconf-0.5.feather; do
  if [ -f "$DIR/$f" ]; then
    echo "skip $f (already downloaded)"
    continue
  fi
  echo "download $f"
  curl -fL --retry 3 -C - -o "$DIR/$f.part" "$BASE/$f"
  mv "$DIR/$f.part" "$DIR/$f"
done
