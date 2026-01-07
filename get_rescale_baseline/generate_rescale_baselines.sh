#!/usr/bin/env bash
set -euo pipefail

# Generate multiple rescale baselines based on get_rescale_baseline/example_run.sh
# Edit the lists below to add/remove models and language/config pairs.

# Hugging Face dataset to draw text from (CulturaX fits many languages)
HF_DATASET="uonlp/culturax"
HF_SPLIT="train"
HF_STREAMING="--hf-streaming"  # set to empty string to disable streaming
# Read exactly this many lines (pairs will be half this). Keep at 1,000,000 to match baseline recipe.
MAX_LINES=100
# If you also want to force the exact number of pairs, set NUM_SAMPLES to half MAX_LINES (comment out to use all).
# NUM_SAMPLES=10000
BATCH_SIZE=32
ENCODERS_FILE="encoders.txt"
LOCAL_DATASET_ROOT="local_datasets"

if [[ ! -f "${ENCODERS_FILE}" ]]; then
  echo "Encoder list not found: ${ENCODERS_FILE}" >&2
  exit 1
fi

# Read model/lang pairs from encoders.txt, skipping blanks and duplicates
MODELS=()
MODEL_LANGS=()
declare -A _seen_pairs=()
while IFS=',' read -r raw_model raw_lang; do
  model=$(echo "${raw_model}" | xargs)
  lang=$(echo "${raw_lang}" | xargs)
  [[ -z "${model}" || -z "${lang}" ]] && continue
  key="${model}:::${lang}"
  [[ -n "${_seen_pairs[$key]:-}" ]] && continue
  _seen_pairs["${key}"]=1
  MODELS+=("${model}")
  MODEL_LANGS+=("${lang}")
done < "${ENCODERS_FILE}"

echo "Generating baselines for dataset=${HF_DATASET}, split=${HF_SPLIT}"

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  lang="${MODEL_LANGS[$idx]}"
  echo ""
  echo ">>> lang=${lang} model=${model}"
  local_dataset_path="${LOCAL_DATASET_ROOT}/${HF_DATASET//\//_}/${lang}"
  local_dataset_flag=()
  local_dataset_flag=(--local-dataset "${local_dataset_path}")
  python get_rescale_baseline/get_rescale_baseline.py \
    --text-field text \
    "${local_dataset_flag[@]}" \
    --max-lines "${MAX_LINES}" \
    -m "${model}" \
    -b "${BATCH_SIZE}"
done
