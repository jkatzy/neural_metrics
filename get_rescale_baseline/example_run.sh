#!/usr/bin/env bash
set -euo pipefail
# Example helper script to run get_rescale_baseline with either a Hugging Face dataset
# or the legacy local-language files. Edit the MODEL and BATCH_SIZE variables below.

MODEL="roberta-large"
BATCH_SIZE=16
LINE_LENGTH_LIMIT=32    

echo "This script runs get_rescale_baseline.py examples.\n"

# If you plan to use a Hugging Face dataset, ensure `datasets` is installed.
if ! python - <<'PY' 2>/dev/null
try:
    import datasets
except Exception:
    raise SystemExit(2)
PY
then
    echo "\nThe 'datasets' package is not available in your Python environment."
    echo "Installing 'datasets' into user site-packages (pip install --user datasets)..."
    pip install --user datasets
fi

echo "\n1) Example: Use a Hugging Face dataset (wikitext)"
echo "   - Dataset id: wikitext"
echo "   - Split: train"
echo "   - Text field: text"

python get_rescale_baseline/get_rescale_baseline.py \
  --hf-dataset uonlp/CulturaX \
  --hf-split train \
  --hf-config en \
  --text-field text \
  -m ${MODEL} -b ${BATCH_SIZE} \
  --line-length-limit ${LINE_LENGTH_LIMIT} \
  --hf-streaming \
  --max-lines 10000