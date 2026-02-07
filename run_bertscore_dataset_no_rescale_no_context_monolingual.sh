#!/usr/bin/env bash
set -euo pipefail

# Driver to run BERTScore over a HF dataset with no rescaling and no context.

DATASET="AISE-TUDelft/multilingual-code-comments-fixed"
SPLIT="train"
REF_FIELD="original_comment"          # can be str or list[str]
MODEL="bert-base-multilingual-cased"
LANG="en"
BATCH_SIZE=64
OUTPUT_DIR="outputs"
ID_FIELD="file_id"
USE_IDF=false     # set to true to enable IDF weighting
ENCODERS_FILE="encoders.txt"

declare -A LANG_TO_CONFIG=(
  [en]="English"
  [pl]="Polish"
  [el]="Greek"
  [zh]="Chinese"
  [nl]="Dutch"
)

mkdir -p "${OUTPUT_DIR}"

while IFS=',' read -r raw_model raw_lang; do
  model=$(echo "${raw_model}" | xargs)
  lang=$(echo "${raw_lang}" | xargs)
  [[ -z "${model}" ]] && continue
  safe_lang=${lang:-${LANG}}
  config=${LANG_TO_CONFIG[${safe_lang}]:-}
  if [[ -z "${config}" ]]; then
    echo "Unknown language code '${safe_lang}' for dataset config" >&2
    exit 1
  fi
  safe_model=${model//\//_}
  output_csv="${OUTPUT_DIR}/"

  echo ">>> Running BERTScore for model=${model} lang=${safe_lang} config=${config} -> ${output_csv}"
  python run_bertscore_dataset.py \
    --dataset "${DATASET}" \
    --config "${config}" \
    --split "${SPLIT}" \
    --ref-field "${REF_FIELD}" \
    --model "${model}" \
    --lang "${safe_lang}" \
    --batch-size "${BATCH_SIZE}" \
    ${ID_FIELD:+--id-field "${ID_FIELD}"} \
    --output-dir "${output_csv}" \
    --return-hash
done < "${ENCODERS_FILE}"
