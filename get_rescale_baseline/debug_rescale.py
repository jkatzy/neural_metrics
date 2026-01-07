#!/usr/bin/env python
"""
Debug helper to invoke get_rescale_baseline.py with parameters you can tweak here
or override via debugger/CLI args. Calls the underlying functions directly
instead of spawning a subprocess, so breakpoints work.
"""
import argparse
import sys
from pathlib import Path
import types

import pandas as pd
import torch
from tqdm.auto import tqdm

# Ensure repo root is importable when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a lightweight sacrebleu stub if not installed (get_rescale_baseline imports it but does not use it)
if "sacrebleu" not in sys.modules:
    sys.modules["sacrebleu"] = types.SimpleNamespace()

import bert_score
from get_rescale_baseline import get_rescale_baseline as grb
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Debug runner for get_rescale_baseline.py")
    p.add_argument("--hf-dataset", default="uonlp/CulturaX")
    p.add_argument("--hf-config", default="zh")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--hf-streaming", action="store_true", default=True)
    p.add_argument("--max-lines", type=int, default=100000)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("-m", "--model", nargs="+", default=["hfl/chinese-macbert-large"])
    p.add_argument("-b", "--batch-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--use-fast-tokenizer", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.model:
        print("Specify at least one model with -m/--model")
        sys.exit(1)

    data_name = args.hf_dataset
    for model_type in args.model:
        # Load data
        hyp, cand = grb.get_data(
            lang=args.hf_dataset,
            split=args.hf_split,
            text_field=args.text_field,
            max_lines=args.max_lines,
            line_length_limit=args.line_length_limit,
            config=args.hf_config,
            streaming=args.hf_streaming,
            num_samples=args.num_samples,
            tokenizer=AutoTokenizer.from_pretrained(model_type, use_fast=args.use_fast_tokenizer),
        )
        baseline_file_path = f"rescale_baseline/{data_name}/{model_type}.tsv"
        print(f"computing baseline for {model_type} on {data_name}")
        scorer = bert_score.BERTScorer(
            model_type=model_type,
            all_layers=True,
            num_layers=args.num_layers,
            use_fast_tokenizer=args.use_fast_tokenizer,
        )
        with torch.no_grad():
            score_means = None
            count = 0
            for batches in tqdm(
                grb.chunk(list(zip(hyp, cand)), 1000), total=len(hyp) / 1000
            ):
                batch_hyp, batch_cand = zip(*batches)
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
        if not Path(baseline_file_path).parent.exists():
            Path(baseline_file_path).parent.mkdir(parents=True, exist_ok=True)
        pd_baselines.to_csv(baseline_file_path)
        del scorer


if __name__ == "__main__":
    main()
