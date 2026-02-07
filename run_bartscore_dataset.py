#!/usr/bin/env python
import argparse
import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from bart_score import BARTScorer

from .utils import replace_with_noise, get_overlap, remove_overlap


def _as_list(x: Union[str, List[str]]) -> List[str]:
    return x if isinstance(x, list) else [x]


def _get_field(record: Dict[str, Any], field: str) -> Any:
    if field is None:
        return None
    if field not in record:
        raise KeyError(f"Field '{field}' not found in dataset record keys: {list(record.keys())}")
    return record[field]


FIM_TOKEN_DICT = {
    "google/codegemma-7b": {"prefix": "<|fim_prefix|>", "middle": "<|fim_middle|>", "suffix": "<|fim_suffix|>"},
    "meta-llama/CodeLlama-7b-hf": {"prefix": "<PRE>", "middle": "<MID>", "suffix": "<SUF>"},
    "Qwen/CodeQwen1.5-7B": {"prefix": "<fim_prefix>", "middle": "<fim_middle>", "suffix": "<fim_suffix>"},
    "bigcode/starcoder2-7b": {"prefix": "<fim_prefix>", "middle": "<fim_middle>", "suffix": "<fim_suffix>"},
    "ibm-granite/granite-8b-code-base": {"prefix": "<fim_prefix>", "middle": "<fim_middle>", "suffix": "<fim_suffix>"},
}


def main():
    parser = argparse.ArgumentParser(
        description="Compute BARTScore for a Hugging Face dataset with per-sample affixes and save to CSV."
    )
    parser.add_argument("--dataset", default="AISE-TUDelft/multilingual-code-comments-fixed", help="HF dataset id or path")
    parser.add_argument("--config", default="English", help="Optional dataset config name")
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument(
        "--llm-models",
        nargs="*",
        default=None,
        help="List of LLM keys to score (defaults to all keys in FIM_TOKEN_DICT). "
             "Each model is expected to have fields predicted_comment_<llm> and masked_data_<llm>.",
    )
    parser.add_argument("--ref-field", default="original_comment", help="Field name for reference text (str or list[str])")
    parser.add_argument("--checkpoint", default="facebook/bart-large-cnn", help="BART checkpoint to load")
    parser.add_argument("--load-path", default=None, help="Optional finetuned weights to load via BARTScorer.load")
    parser.add_argument("--device", default=None, help="Device string for BARTScorer (defaults to cuda if available)")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length for tokenizer/model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for scoring")
    parser.add_argument(
        "--use-context",
        default=None,
        choices=['none', 'minimal', 'full'],
        help="If set, keep affixes in the text (do NOT strip prefixes/suffixes before scoring). Default strips them.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/debug_bartscore.csv",
        # required=True,
        help="Path to save CSV with columns: cand, refs (JSON), forward, reverse, and averaged BARTScores.",
    )
    parser.add_argument("--id-field", default="file_id", help="Field name for a sample identifier (saved to CSV).")
    parser.add_argument(
        "--noise-type",
        default=None,
        choices=["uniform", "targeted"],
        help="If set, replace candidate text with noise of the specified type before scoring. "
             "'uniform' replaces with random tokens from the vocabulary, 'targeted' replaces with tokens from the context. Default is no noise.",
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split)
    llm_models = args.llm_models or list(FIM_TOKEN_DICT.keys())

    ids: List[Any] = []
    cands: List[str] = []
    refs: List[Union[str, List[str]]] = []
    cand_prefixes: List[str] = []
    cand_suffixes: List[str] = []
    ref_prefixes: List[Union[str, List[str]]] = []
    ref_suffixes: List[Union[str, List[str]]] = []
    llm_labels: List[str] = []

    for rec in ds:
        base_id = _get_field(rec, args.id_field) if args.id_field else len(ids)
        ref = _get_field(rec, args.ref_field)
        if ref is None:
            raise ValueError("Reference field missing for a record")

        for llm in llm_models:
            tokens = FIM_TOKEN_DICT[llm]
            cand_field = f"predicted_comment_{llm}"
            context_field = f"masked_data_{llm}"
            cand = _get_field(rec, cand_field)
            context = _get_field(rec, context_field)
            if cand is None or context is None:
                context = ""
                cand = ""
                # raise ValueError(f"Candidate or context field missing for LLM '{llm}'")

            ref = ref.replace("_x000D_", "")
            cand = cand.replace("_x000D_", "")
            context = context.replace("_x000D_", "")

            split_ctx = context.split(tokens["suffix"])
            if len(split_ctx) < 2:
                split_ctx = ["",""]
                # raise ValueError(f"Context for LLM '{llm}' does not contain suffix token")
            cand_pre = split_ctx[0].replace(tokens["prefix"], "")
            cand_suf = split_ctx[1].replace(tokens["middle"], "")

            if args.use_context == 'minimal':
                cand_pre = get_overlap(cand_pre, ref)
                cand_suf = ""

            ref_pre = cand_pre
            ref_suf = cand_suf

            ref_list = _as_list(ref)
            if isinstance(ref_pre, list):
                if len(ref_pre) != len(ref_list):
                    raise ValueError("ref_prefix_field length does not match number of references")
                ref_pre_list = ref_pre
            else:
                ref_pre_list = [ref_pre] * len(ref_list)

            if isinstance(ref_suf, list):
                if len(ref_suf) != len(ref_list):
                    raise ValueError("ref_suffix_field length does not match number of references")
                ref_suf_list = ref_suf
            else:
                ref_suf_list = [ref_suf] * len(ref_list)

            if args.noise_type == 'uniform' or args.noise_type == 'targeted':
                cand = replace_with_noise(cand, args.noise_type, args.checkpoint, context)
            
            cands.append(cand)
            ref = remove_overlap(ref_pre, ref)

            refs.append(ref_list if len(ref_list) > 1 else ref_list[0])
            cand_prefixes.append(cand_pre)
            cand_suffixes.append(cand_suf)
            ref_prefixes.append(ref_pre_list if len(ref_pre_list) > 1 else ref_pre_list[0])
            ref_suffixes.append(ref_suf_list if len(ref_suf_list) > 1 else ref_suf_list[0])
            ids.append(base_id)
            llm_labels.append(llm)

    if not args.use_context or args.use_context == 'none':
        cand_prefixes = ["" for _ in cands]
        cand_suffixes = ["" for _ in cands]
        ref_prefixes = [([""] * len(r) if isinstance(r, list) else "") for r in refs]
        ref_suffixes = [([""] * len(r) if isinstance(r, list) else "") for r in refs]

    device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    bart_scorer = BARTScorer(device=device, max_length=args.max_length, checkpoint=args.checkpoint)
    if args.load_path:
        bart_scorer.load(args.load_path)

    forward_scores = bart_scorer.score(
        cands,
        refs,
        batch_size=args.batch_size,
        prefixes_ref=ref_prefixes,
        suffixes_ref=ref_suffixes,
        prefixes_cand=cand_prefixes,
        suffixes_cand=cand_suffixes,
    )
    reverse_scores = bart_scorer.score(
        refs,
        cands,
        batch_size=args.batch_size,
        prefixes_ref=cand_prefixes,
        suffixes_ref=cand_suffixes,
        prefixes_cand=ref_prefixes,
        suffixes_cand=ref_suffixes,
    )
    avg_scores = 0.5 * (np.array(forward_scores) + np.array(reverse_scores))

    df = pd.DataFrame(
        {
            "id": ids,
            "llm": llm_labels,
            "bart_score_forward": forward_scores,
            "bart_score_reverse": reverse_scores,
            "bart_score_avg": avg_scores,
        }
    )

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    avg_forward = float(np.mean(forward_scores))
    avg_reverse = float(np.mean(reverse_scores))
    avg_avg = float(np.mean(avg_scores))
    print(f"Average BARTScore -> forward: {avg_forward:.6f} reverse: {avg_reverse:.6f} avg: {avg_avg:.6f}")


if __name__ == "__main__":
    main()
