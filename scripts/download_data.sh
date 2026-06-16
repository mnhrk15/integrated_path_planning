#!/usr/bin/env bash
# Download ETH/UCY pedestrian trajectory datasets (SGAN distribution).
# Extracts to ./datasets/. Raw data is gitignored (.gitignore: datasets/, *.zip)
# and must NOT be committed/re-distributed (datasets are research-use, cite-only).
set -euo pipefail

URL='https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=1'

if command -v wget >/dev/null 2>&1; then
  wget -O datasets.zip "$URL"
else
  curl -L -o datasets.zip "$URL"
fi

unzip -q datasets.zip
rm -f datasets.zip
