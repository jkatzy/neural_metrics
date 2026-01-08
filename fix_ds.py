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
        comment = parts[1]
        if '<file_sep>' in comment:
            comment = comment.split('<file_sep>')[0]
        if '<|file_separator|>' in comment:
            comment = comment.split('<|file_separator|>')[0]
        if '<eos>' in comment:
            comment = comment.split('<eos>')[0]
        ex[f"predicted_comment_{llm_name}"] = comment
    return ex


for lang in LANGS:
    ds = load_dataset("AISE-TUDelft/multilingual-code-comments", lang, split="train")
    ds = ds.map(extract_comment, num_proc=os.cpu_count() or 1)
    ds.push_to_hub("AISE-TUDelft/multilingual-code-comments-fixed", config_name=lang, split="train")

