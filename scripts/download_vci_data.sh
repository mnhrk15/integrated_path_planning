#!/usr/bin/env bash
# Place & extract the VCI DUT/CITR vehicle-crowd interaction datasets.
#
# These datasets are distributed via Google Drive / Baidu Yun (NOT a direct
# download URL), have NO explicit license (research-use; cite IEEE IV2019,
# arXiv:1902.00487), and MUST NOT be committed or redistributed
# (.gitignore excludes datasets/).
#
# Manual step (cannot be automated -- Drive requires interactive download):
#   1. Open the dataset README links and download the archives:
#        DUT:  https://github.com/dongfang-steven-yang/vci-dataset-dut
#        CITR: https://github.com/dongfang-steven-yang/vci-dataset-citr
#   2. Save them into datasets/ as:
#        datasets/vci-dataset-dut.zip
#        datasets/vci-dataset-citr.zip
#   3. Run this script to extract into datasets/vci_dut/ and datasets/vci_citr/.
set -euo pipefail

mkdir -p datasets/vci_dut datasets/vci_citr

extract() {
  local zip="$1" dest="$2"
  if [[ -f "$zip" ]]; then
    echo "Extracting $zip -> $dest"
    unzip -q -o "$zip" -d "$dest"
  else
    echo "MISSING: $zip"
    echo "  Download it manually from the Google Drive link in the dataset"
    echo "  README, save it as $zip, then re-run this script."
  fi
}

extract datasets/vci-dataset-dut.zip datasets/vci_dut
extract datasets/vci-dataset-citr.zip datasets/vci_citr
