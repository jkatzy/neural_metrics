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

import re

def clean_eos_tokens(text: str) -> str:
    """
    Remove end-of-sequence tokens and everything after them.
    
    Handles tokens like:
    - <file_sep>
    - <|file_separator|>
    - <|endoftext|>
    - <fim_suffix>
    - <fim_middle>
    - </s>
    - And other common EOS tokens
    """
    if not text:
        return text
    
    # Define EOS tokens (order matters - check longer patterns first)
    eos_patterns = [
        r'<\|file_separator\|>',
        r'<\|endoftext\|>',
        r'<file_sep>',
        r'<fim_suffix>',
        r'<fim_middle>',
        r'<fim_pad>',
        r'<\|fim_suffix\|>',
        r'<\|fim_middle\|>',
        r'</s>',
        r'<eos>',
        r'<\|end\|>',
    ]
    
    # Find the earliest EOS token
    earliest_pos = len(text)
    matched_token = None
    
    for pattern in eos_patterns:
        match = re.search(pattern, text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            matched_token = pattern
    
    # Truncate at the EOS token
    if earliest_pos < len(text):
        text = text[:earliest_pos]
    
    # Clean up trailing whitespace
    return text.rstrip()

def extract_comment(ex):
    for field, llm_name, middle_token in _LLM_INFO:
        generation = ex[field]
        parts = generation.split(middle_token, 1)
        if len(parts) != 2:
            print(f"Middle token '{middle_token}' missing in generation for {llm_name}")
            continue
        comment = parts[1]
        comment = clean_eos_tokens(comment)

        ex[f"predicted_comment_{llm_name}"] = comment
    return ex




for lang in LANGS:
    ds = load_dataset("AISE-TUDelft/multilingual-code-comments", lang, split="train")
    ds = ds.map(extract_comment, num_proc=os.cpu_count() or 1)
    ds.push_to_hub("AISE-TUDelft/multilingual-code-comments-fixed", config_name=lang, split="train")

