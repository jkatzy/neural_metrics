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
CONTEXT_SETTINGS=("none" "minimal" "full")  # none/minimal/full
NOISE_SETTINGS=("none" "targeted" "uniform") # none/targeted/uniform
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
  for context_setting in "${CONTEXT_SETTINGS[@]}"; do
    for noise_setting in "${NOISE_SETTINGS[@]}"; do
      safe_model=${model//\//_}
      safe_dataset=$(slugify "${DATASET}")
      safe_split=$(slugify "${SPLIT}")
      safe_ref=$(slugify "${REF_FIELD}")
      safe_load=$(slugify "${LOAD_PATH:-none}")
      safe_device=$(slugify "${DEVICE:-auto}")
      safe_llms=$(slugify "${LLM_MODELS[*]:-all}")
      safe_id=$(slugify "${ID_FIELD:-none}")
      safe_context=$(slugify "${context_setting}")
      safe_noise=$(slugify "${noise_setting}")
      settings="_ctx-${safe_context}_noise-${safe_noise}"
      output_csv="${OUTPUT_DIR}/bartscores_lang-${lang}_model-${safe_model}_${settings}.csv"

      echo ">>> Running BARTScore for model=${model} lang=${lang} config=${config} ctx=${context_setting} noise=${noise_setting} -> ${output_csv}"
      echo python run_bartscore_dataset.py \
        --dataset "${DATASET}" \
        --config "${config}" \
        --split "${SPLIT}" \
        --ref-field "${REF_FIELD}" \
        --checkpoint "${model}" \
        --device "${DEVICE}" \
        --max-length "${MAX_LENGTH}" \
        --batch-size "${BATCH_SIZE}" \
        --id-field "${ID_FIELD}" \
        --use-context "${context_setting}" \
        --noise-type "${noise_setting}" \
        --output-csv "${output_csv}"
    done
  done
done < "${ENCODERS_FILE}"
