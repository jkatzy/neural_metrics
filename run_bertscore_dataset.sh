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
OUTPUT_CSV="outputs/bertscores.csv"
ID_FIELD="file_id"
USE_CONTEXT=false  # set to true to keep affixes in the text (no stripping)
USE_IDF=false     # set to true to enable IDF weighting
LLM_MODELS=()     # leave empty to use all keys in FIM_TOKEN_DICT

python run_bertscore_dataset.py \
  --dataset "${DATASET}" \
  ${CONFIG:+--config "${CONFIG}"} \
  --split "${SPLIT}" \
  --ref-field "${REF_FIELD}" \
  ${LLM_MODELS:+--llm-models ${LLM_MODELS[@]}} \
  --model "${MODEL}" \
  --lang "${LANG}" \
  --batch-size "${BATCH_SIZE}" \
  ${ID_FIELD:+--id-field "${ID_FIELD}"} \
  --output-csv "${OUTPUT_CSV}" \
  --return-hash \
  # --use-context
  # --idf ${USE_IDF} \
  # --use-context ${USE_CONTEXT}
