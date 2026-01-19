#!/usr/bin/env bash
set -euo pipefail

# Example driver to run BARTScore over a HF dataset that carries text and affix fields.
# Edit the variables below to match your dataset schema and model setup.

DATASET="AISE-TUDelft/multilingual-code-comments"
SPLIT="train"
REF_FIELD="original_comment"          # can be str or list[str]
LOAD_PATH=""                          # optional path to finetuned weights (leave empty to skip)
DEVICE=""                             # optional device override, e.g., cuda:0
MAX_LENGTH=1024
BATCH_SIZE=4
OUTPUT_DIR="outputs"
ID_FIELD="file_id"
USE_CONTEXT=true  # set to true to keep affixes in the text (no stripping)
LLM_MODELS=()     # leave empty to use all keys in FIM_TOKEN_DICT
ENCODERS_FILE="encoder-decoders.txt"

declare -A LANG_TO_CONFIG=(
  [en]="English"
  [pl]="Polish"
  [el]="Greek"
  [zh]="Chinese"
  [nl]="Dutch"
)

mkdir -p "${OUTPUT_DIR}"

slugify() {
  echo "$1" | tr ' /:' '_' | tr -cd '[:alnum:]_.-'
}

while IFS=',' read -r raw_model raw_lang; do
  model=$(echo "${raw_model}" | xargs)
  lang=$(echo "${raw_lang}" | xargs)
  [[ -z "${model}" ]] && continue
  config=${LANG_TO_CONFIG[${lang}]:-}
  if [[ -z "${config}" ]]; then
    echo "Unknown language code '${lang}' for dataset config" >&2
    exit 1
  fi
  safe_model=${model//\//_}
  safe_dataset=$(slugify "${DATASET}")
  safe_split=$(slugify "${SPLIT}")
  safe_ref=$(slugify "${REF_FIELD}")
  safe_load=$(slugify "${LOAD_PATH:-none}")
  safe_device=$(slugify "${DEVICE:-auto}")
  safe_llms=$(slugify "${LLM_MODELS[*]:-all}")
  safe_id=$(slugify "${ID_FIELD:-none}")
  safe_context=$(slugify "${USE_CONTEXT}")
  settings="ds-${safe_dataset}_cfg-${config}_split-${safe_split}_ref-${safe_ref}_load-${safe_load}_dev-${safe_device}_max-${MAX_LENGTH}_bs-${BATCH_SIZE}_ctx-${safe_context}_id-${safe_id}_llms-${safe_llms}"
  output_csv="${OUTPUT_DIR}/bartscores_${lang}_${safe_model}_${settings}.csv"

  echo ">>> Running BARTScore for model=${model} lang=${lang} config=${config} -> ${output_csv}"
  python run_bartscore_dataset.py \
    --dataset "${DATASET}" \
    --config "${config}" \
    --split "${SPLIT}" \
    --ref-field "${REF_FIELD}" \
    ${LLM_MODELS:+--llm-models ${LLM_MODELS[@]}} \
    --checkpoint "${model}" \
    ${LOAD_PATH:+--load-path "${LOAD_PATH}"} \
    ${DEVICE:+--device "${DEVICE}"} \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    ${ID_FIELD:+--id-field "${ID_FIELD}"} \
    --output-csv "${output_csv}" \
    # $([ "${USE_CONTEXT}" = true ] && echo "--use-context")
done < "${ENCODERS_FILE}"
