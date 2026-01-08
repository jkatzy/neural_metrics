#!/usr/bin/env bash
set -euo pipefail

# Example driver to run BERTScore over a HF dataset that carries text and affix fields.
# Edit the variables below to match your dataset schema.

DATASET="AISE-TUDelft/multilingual-code-comments-fixed"
CONFIG="English"
SPLIT="train"
REF_FIELD="original_comment"          # can be str or list[str]
MODEL="bert-base-multilingual-cased"
LANG="en"
BATCH_SIZE=16
OUTPUT_DIR="outputs"
ID_FIELD="file_id"
USE_CONTEXT=false  # set to true to keep affixes in the text (no stripping)
USE_IDF=false     # set to true to enable IDF weighting
LLM_MODELS=()     # leave empty to use all keys in FIM_TOKEN_DICT
ENCODERS_FILE="encoders.txt"

mkdir -p "${OUTPUT_DIR}"

while IFS=',' read -r raw_model raw_lang; do
  model=$(echo "${raw_model}" | xargs)
  lang=$(echo "${raw_lang}" | xargs)
  [[ -z "${model}" ]] && continue
  safe_model=${model//\//_}
  safe_lang=${lang:-${LANG}}
  output_csv="${OUTPUT_DIR}/bertscores_${safe_lang}_${safe_model}.csv"

  echo ">>> Running BERTScore for model=${model} lang=${safe_lang} -> ${output_csv}"
  python run_bertscore_dataset.py \
    --dataset "${DATASET}" \
    ${CONFIG:+--config "${CONFIG}"} \
    --split "${SPLIT}" \
    --ref-field "${REF_FIELD}" \
    ${LLM_MODELS:+--llm-models ${LLM_MODELS[@]}} \
    --model "${model}" \
    --lang "${safe_lang}" \
    --batch-size "${BATCH_SIZE}" \
    ${ID_FIELD:+--id-field "${ID_FIELD}"} \
    --output-csv "${output_csv}" \
    --return-hash \
    --use-context
    # --idf ${USE_IDF} \
    # --use-context ${USE_CONTEXT}
done < "${ENCODERS_FILE}"
