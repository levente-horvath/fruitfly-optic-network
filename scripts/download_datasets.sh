#!/usr/bin/env bash
# Download Fashion-MNIST (~30 MB) and MNIST (~11 MB) into data/datasets/.
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)/data/datasets"
FILES="train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz"

fetch() { # fetch <base url> <dataset name>
  mkdir -p "$DIR/$2"
  for f in $FILES; do
    if [ -f "$DIR/$2/$f" ]; then
      echo "skip $2/$f (already downloaded)"
      continue
    fi
    echo "download $2/$f"
    curl -fL --retry 3 -o "$DIR/$2/$f.part" "$1/$f"
    mv "$DIR/$2/$f.part" "$DIR/$2/$f"
  done
}

fetch https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion fashion-mnist
fetch https://ossci-datasets.s3.amazonaws.com/mnist mnist
