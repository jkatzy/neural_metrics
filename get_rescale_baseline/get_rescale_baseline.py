import argparse
import gzip
import os
import random
from random import shuffle
import gc

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import bert_score
from transformers import AutoTokenizer


def get_data(
    lang,
    split="train",
    text_field=None,
    max_lines=1_000_000,
    line_length_limit=32,
    config=None,
    streaming=True,
    num_samples=None,
    tokenizer=None,
    local_dataset=None,
):
    """Load a HF or local dataset and return randomly paired hyp/cand lists."""

    from datasets import load_dataset, load_from_disk

    if local_dataset:
        loaded = load_from_disk(local_dataset)
        ds = loaded[split] if hasattr(loaded, "keys") and split in loaded else loaded
        streaming = False
    else:
        try:
            ds = load_dataset(lang, config, split=split, streaming=streaming)
        except Exception:
            if streaming:
                ds = load_dataset(lang, config, split=split, streaming=False)
                streaming = False
            else:
                raise
    lines = []

    def push_line(s):
        s = s.strip()
        toks = tokenizer.tokenize(s) if tokenizer is not None else s.split()
        if not toks:
            return
        if len(toks) > line_length_limit:
            start = random.randint(0, len(toks) - line_length_limit)
            toks = toks[start : start + line_length_limit]
        text_out = (
            tokenizer.convert_tokens_to_string(toks)
            if tokenizer is not None
            else " ".join(toks)
        ).strip()
        if not text_out:
            return
        token_check = (
            tokenizer.encode(text_out, add_special_tokens=False)
            if tokenizer is not None
            else text_out.split()
        )
        if len(token_check) == 0:
            return
        lines.append(text_out)

    ds_iter = iter(ds)
    # infer text field from first example
    try:
        first = next(ds_iter)
    except StopIteration:
        raise ValueError("Dataset is empty")
    if text_field is None and isinstance(first, dict):
        if "text" in first:
            text_field = "text"
        else:
            for k, v in first.items():
                if isinstance(v, str):
                    text_field = k
                    break

    count = 0
    # include first record
    if text_field and isinstance(first, dict) and text_field in first:
        push_line(str(first[text_field]))
    elif isinstance(first, str):
        push_line(first)
    count += 1

    for ex in ds_iter:
        if count >= max_lines:
            break
        if text_field and isinstance(ex, dict) and text_field in ex:
            push_line(str(ex[text_field]))
        elif isinstance(ex, str):
            push_line(ex)
        count += 1

    if len(lines) < 2:
        raise ValueError("Not enough lines collected to produce hyp/cand pairs")

    pair_count = num_samples if num_samples is not None else len(lines) // 2
    if pair_count > len(lines) // 2:
        raise ValueError("Requested more samples than available pairs")

    samples = np.random.choice(range(len(lines)), size=(2, pair_count), replace=False)
    hyp = [lines[i] for i in samples[0]]
    cand = [lines[i] for i in samples[1]]
    return hyp, cand


def chunk(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="language to compute baseline with (legacy local files)."
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Hugging Face dataset id or path (overrides --lang)."
    )
    parser.add_argument(
        "--local-dataset",
        type=str,
        default=None,
        help="Path to a local dataset saved with datasets.save_to_disk (overrides --hf-dataset).",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default=None,
        help="(Optional) Hugging Face dataset config name (passed as 'name' to load_dataset)."
    )
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        help="Load the Hugging Face dataset in streaming mode (IterableDataset).",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Split to use when loading a HF dataset (default: train)."
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="Name of the text field/column in the dataset (optional)."
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=1000000,
        help="Maximum number of lines/examples to read (default: 1_000_000)."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of hyp/ref pairs to score (default: use all available pairs).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Optional num_layers override for the model (helps unknown models).",
    )
    parser.add_argument(
        "--use-fast-tokenizer",
        action="store_false",
        help="Use HF fast tokenizer (set this flag to disable fast tokenizers).",
    )
    parser.add_argument(
        "--line-length-limit",
        type=int,
        default=32,
        help="Maximum number of tokens per line to include in baseline data (default: 32)."
    )
    parser.add_argument("-m", "--model", nargs="+", help="models to tune")
    parser.add_argument("-b", "--batch_size", type=int, default=64)

    args = parser.parse_args()
    
    # Require a dataset source
    if not args.hf_dataset and not args.local_dataset:
        raise SystemExit("Please specify --hf-dataset or --local-dataset")
    if not args.model:
        raise SystemExit("Please specify at least one model with -m/--model")
    
    args.use_fast_tokenizer = False
    
    def _safe_name(name: str) -> str:
        return name.replace("/", "_")

    data_name = _safe_name(args.hf_dataset) if args.hf_dataset else _safe_name(args.local_dataset)
    config_label = args.hf_config if args.hf_config else "default"
    baseline_dir = f"rescale_baseline/{data_name}/{config_label}"
    # Determine which baselines need computing before loading any dataset to avoid
    # unnecessary work (and potential crashes) when all outputs already exist.
    todo_models = []
    for model_type in args.model:
        baseline_file_path = f"{baseline_dir}/{model_type}.tsv"
        if os.path.isfile(baseline_file_path):
            print(f"{model_type} baseline exists for {data_name} ({config_label})")
        else:
            todo_models.append(model_type)

    if not todo_models:
        raise SystemExit("All requested baselines already exist; nothing to do.")

    def _validate_batch(texts, tokenizer):
        vocab_size = tokenizer.vocab_size
        max_len = tokenizer.model_max_length
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=True)
            if not ids:
                raise ValueError("Tokenization yielded empty ids")
            if max(ids) >= vocab_size or min(ids) < 0:
                raise ValueError(f"Token ids out of bounds for vocab size {vocab_size}")
            if len(ids) > max_len:
                raise ValueError(f"Tokenized length {len(ids)} exceeds model_max_length {max_len}")

    for model_type in todo_models:
        baseline_file_path = f"{baseline_dir}/{model_type}.tsv"
        print(f"computing baseline for {model_type} on {data_name} ({config_label})")
        scorer = bert_score.BERTScorer(
            model_type=model_type,
            all_layers=True,
            num_layers=args.num_layers,
            use_fast_tokenizer=args.use_fast_tokenizer,
        )

        hyp, cand = get_data(
            lang=args.hf_dataset or args.local_dataset,
            split=args.hf_split,
            text_field=args.text_field,
            max_lines=args.max_lines,
            line_length_limit=args.line_length_limit,
            config=args.hf_config,
            streaming=args.hf_streaming,
            num_samples=args.num_samples,
            tokenizer=AutoTokenizer.from_pretrained(model_type, use_fast=args.use_fast_tokenizer),
            local_dataset=args.local_dataset,
        )
        with torch.no_grad():
            score_means = None
            count = 0
            for batches in tqdm(
                chunk(list(zip(hyp, cand)), 1000), total=len(hyp) / 1000
            ):
                batch_hyp, batch_cand = zip(*batches)
                _validate_batch(batch_hyp, scorer._tokenizer)
                _validate_batch(batch_cand, scorer._tokenizer)
                scores = scorer.score(
                    batch_hyp, batch_cand, batch_size=args.batch_size
                )
                scores = torch.stack(scores, dim=0)
                if score_means is None:
                    score_means = scores.mean(dim=-1)
                else:
                    score_means = score_means * count / (
                        count + len(batches)
                    ) + scores.mean(dim=-1) * len(batches) / (count + len(batches))
                count += len(batches)

        pd_baselines = pd.DataFrame(
            score_means.numpy().transpose(), columns=["P", "R", "F"]
        )
        pd_baselines.index.name = "LAYER"

        os.makedirs(os.path.dirname(baseline_file_path), exist_ok=True)
        pd_baselines.to_csv(baseline_file_path)
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
