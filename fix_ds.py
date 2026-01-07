import os

from datasets import load_dataset
from run_bertscore_dataset import FIM_TOKEN_DICT

LANGS = ["English", "Chinese", "Dutch", "Polish", "Greek"]
LLM_FIELDS = [
    "predict_bigcode/starcoder2-7b",
    "predict_Qwen/CodeQwen1.5-7B",
    "predict_ibm-granite/granite-8b-code-base",
    "predict_google/codegemma-7b",
    "predict_meta-llama/CodeLlama-7b-hf",
]

_LLM_INFO = [
    (
        field,
        field.replace("predict_", ""),
        FIM_TOKEN_DICT[field.replace("predict_", "")]["middle"],
    )
    for field in LLM_FIELDS
]


def extract_comment(ex):
    for field, llm_name, middle_token in _LLM_INFO:
        generation = ex[field]
        parts = generation.split(middle_token, 1)
        if len(parts) != 2:
            print(f"Middle token '{middle_token}' missing in generation for {llm_name}")
            continue
        ex[f"predicted_comment_{llm_name}"] = parts[1]
    return ex


for lang in LANGS:
    ds = load_dataset("AISE-TUDelft/multilingual-code-comments-fixed", lang, split="train")
    ds = ds.map(extract_comment, num_proc=os.cpu_count() or 1)
    # ds.push_to_hub("AISE-TUDelft/multilingual-code-comments-fixed", config_name=lang, split="train")
