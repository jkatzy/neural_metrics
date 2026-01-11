#!/usr/bin/env python
import argparse
import json
import os
from typing import Any, Dict, List, Union

import pandas as pd
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from bert_score.utils import get_tokenizer

import bert_score


def _as_list(x: Union[str, List[str]]) -> List[str]:
    return x if isinstance(x, list) else [x]


def _get_field(record: Dict[str, Any], field: str) -> Any:
    if field is None:
        return None
    if field not in record:
        raise KeyError(f"Field '{field}' not found in dataset record keys: {list(record.keys())}")
    if record[field] is None:
        print(f"record entry is None for {field}")
        return ""
    return record[field]


def _sanitize_hash_code(hash_code: str) -> str:
    if not hash_code:
        return hash_code
    for sep in {"/", os.sep, os.altsep}:
        if sep:
            hash_code = hash_code.replace(sep, "__")
    return hash_code


def _validate_sequences(
    texts: List[Union[str, List[str]]], tokenizer, label: str, max_positions: int = None
) -> None:
    vocab_size = tokenizer.vocab_size
    max_len = tokenizer.model_max_length
    idx = 0
    for item in texts:
        seqs = item if isinstance(item, list) else [item]
        for t in seqs:
            ids = tokenizer.encode(t, add_special_tokens=True)
            if not ids:
                raise ValueError(f"{label}[{idx}] tokenizes to empty ids")
            # if min(ids) < 0 or max(ids) >= vocab_size:
            #     raise ValueError(
            #         f"{label}[{idx}] has token id out of bounds (vocab_size={vocab_size})"
            #     )
            if len(ids) > max_len:
                raise ValueError(
                    f"{label}[{idx}] tokenized length {len(ids)} exceeds model_max_length {max_len}"
                )
            if max_positions is not None and len(ids) > max_positions:
                raise ValueError(
                    f"{label}[{idx}] tokenized length {len(ids)} exceeds model max_position_embeddings {max_positions}"
                )
        idx += 1

FIM_TOKEN_DICT = {'google/codegemma-7b': {'prefix': '<|fim_prefix|>', 'middle': '<|fim_middle|>', 'suffix': '<|fim_suffix|>'},
                  'meta-llama/CodeLlama-7b-hf' :{'prefix': '<PRE>', 'middle': '<MID>', 'suffix': '<SUF>'},
                  'Qwen/CodeQwen1.5-7B' :{'prefix': '<fim_prefix>', 'middle': '<fim_middle>', 'suffix': '<fim_suffix>'},
                  'bigcode/starcoder2-7b' :{'prefix': '<fim_prefix>', 'middle': '<fim_middle>', 'suffix': '<fim_suffix>'},
                  'ibm-granite/granite-8b-code-base' :{'prefix': '<fim_prefix>', 'middle': '<fim_middle>', 'suffix': '<fim_suffix>'}}


def remove_overlap(a, b):
    max_overlap = 0
    max_len = min(len(a), len(b))

    for i in range(1, max_len + 1):
        if a[-i:] == b[:i]:
            max_overlap = i

    return b[max_overlap:]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute BERTScore for a Hugging Face dataset with per-sample affixes and save to CSV."
    )
    parser.add_argument("--dataset", default=None, help="HF dataset id or path")
    parser.add_argument("--config", default=None, help="Optional dataset config name")
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument(
        "--llm-models",
        nargs="*",
        default=None,
        help="List of LLM keys to score (defaults to all keys in FIM_TOKEN_DICT). "
             "Each model is expected to have fields predicted_comment_<llm> and masked_data_<llm>.",
    )
    parser.add_argument("--ref-field", default=None, help="Field name for reference text (str or list[str])")
    parser.add_argument("--model", default=None, help="Model to use (passes through to bert_score.score)")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers (optional)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for scoring")
    parser.add_argument("--idf", action="store_true", help="Use IDF weighting")
    parser.add_argument("--lang", default=None, help="Language code (required if rescaling)")
    parser.add_argument("--rescale-with-baseline", action="store_true", help="Rescale scores with baseline")
    parser.add_argument("--baseline-path", default=None, help="Optional custom baseline path")
    parser.add_argument(
        "--use-fast-tokenizer",
        action="store_true",
        help="Use HF fast tokenizer (default: slow tokenizer).",
    )
    parser.add_argument("--id-field", default=None, help="Field name for a sample identifier (saved to CSV).")
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="If true, adds context to the generted comments before scoring, default True.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        default="./outputs/",
        help="Directory to save CSV with columns: cand, refs (JSON), P, R, F, filename is the hash",
    )
    parser.add_argument("--return-hash", action="store_true", help="Include hash code column in output")
    parser.add_argument("--multilingual", required=False, action="store_false", default="False", help=" Set to True if running a multilingual model, only used for hash and saving, default False")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.return_hash = True
    # Fallback defaults to allow running without CLI args (handy for debugger).
    if args.dataset is None:
        args.dataset = "AISE-TUDelft/multilingual-code-comments"
    if args.ref_field is None:
        args.ref_field = "original_comment"
    if args.model is None:
        args.model = "novelcore/gem-modernbert"
    # if args.output_csv is None:
    #     args.output_csv = "outputs/bertscores.csv"
    if args.config is None:
        args.config = "Greek"
    # args.rescale_with_baseline = True
    model_config = AutoConfig.from_pretrained(args.model)
    validation_tokenizer = get_tokenizer(args.model)

    # args.lang = "el"
    # args.use_context = True
    # if args.use_context is None:
    # args.use_context = True

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

        for llm in llm_models:
            ref = _get_field(rec, args.ref_field)
            tokens = FIM_TOKEN_DICT[llm]
            cand_field = f"predicted_comment_{llm}"
            context_field = f"masked_data_{llm}"
            cand = _get_field(rec, cand_field)
            context = _get_field(rec, context_field)
            if cand is None or context is None:
                raise ValueError(f"Candidate or context field missing for LLM '{llm}'")

            split_ctx = context.split(tokens["suffix"])
            if len(split_ctx) < 2:
                split_ctx=["",""]
                # raise ValueError(f"Context for LLM '{llm}' does not contain suffix token")
            cand_pre = split_ctx[0].replace(tokens["prefix"], "")
            cand_suf = split_ctx[1].replace(tokens["middle"], "")

            ref_pre = cand_pre
            ref_suf = cand_suf

            ref = remove_overlap(ref_pre, ref)

            # Normalize references and affixes to list-of-strings
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

            cands.append(cand)
            refs.append(ref_list if len(ref_list) > 1 else ref_list[0])
            cand_prefixes.append(cand_pre)
            cand_suffixes.append(cand_suf)
            ref_prefixes.append(ref_pre_list if len(ref_pre_list) > 1 else ref_pre_list[0])
            ref_suffixes.append(ref_suf_list if len(ref_suf_list) > 1 else ref_suf_list[0])
            ids.append(base_id)
            llm_labels.append(llm)

    if not args.use_context:
        # If not using context, clear all affixes, not very efficient but will not kill us, easy to implement here
        cand_prefixes = [""] * len(cand_prefixes)
        cand_suffixes = [""] * len(cand_suffixes)
        ref_prefixes = [""] * len(ref_prefixes)
        ref_suffixes = [""] * len(ref_suffixes)

    # Call bert_score.score with per-item affixes
    result = bert_score.score(
        cands,
        refs,
        model_type=args.model,
        num_layers=args.num_layers,
        idf=args.idf,
        batch_size=args.batch_size,
        lang=args.lang,
        rescale_with_baseline=args.rescale_with_baseline,
        baseline_path=args.baseline_path,
        use_fast_tokenizer=args.use_fast_tokenizer,
        use_context=args.use_context,
        multilingual=args.multilingual,
        strip_prefix_ref=ref_prefixes,
        strip_suffix_ref=ref_suffixes,
        strip_prefix_cand=cand_prefixes,
        strip_suffix_cand=cand_suffixes,
        return_hash=args.return_hash,
    )
    if args.return_hash:
        scores, hash_code = result
    else:
        scores, hash_code = result, None
    P, R, F = scores
    df = pd.DataFrame(
        {
            "id": ids,
            "llm": llm_labels,
            "P": P.tolist(),
            "R": R.tolist(),
            "F": F.tolist(),
        }
    )
    if args.return_hash and hash_code is not None:
        hash_code = _sanitize_hash_code(hash_code)
        df["hash"] = hash_code

    out_dir = os.path.dirname(args.output_dir)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, hash_code) + ".csv", index=False)

    avg_P = float(P.mean())
    avg_R = float(R.mean())
    avg_F = float(F.mean())
    print(f"Average BERTScore -> P: {avg_P:.6f} R: {avg_R:.6f} F: {avg_F:.6f}")


if __name__ == "__main__":
    main()
