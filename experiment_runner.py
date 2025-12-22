"""
CLI helper for running BERTScore experiments on HF datasets.

This adapts the notebook driver logic into a reusable script that:
- loads a dataset split
- computes BERTScore columns for one or more model checkpoints
- prints mean P/R/F1 per checkpoint
"""
import argparse
import os
import re
from typing import Iterable, List, Tuple

import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import login

from custom_bertscore import score as bert_score
from custom_bartscore import score as bart_score

FIM_TOKEN_DICT = {'google/codegemma-7b': {'prefix': '<|fim_prefix|>', 'middle': '<|fim_middle|>', 'suffix': '<|fim_suffix|>'},
                  'meta-llama/CodeLlama-7b-hf' :{'prefix': '<PRE>', 'middle': '<MID>', 'suffix': '<SUF>'},
                  'Qwen/CodeQwen1.5-7B' :{'prefix': '<fim_prefix>', 'middle': '<fim_middle>', 'suffix': '<fim_suffix>'},
                  'bigcode/starcoder2-7B' :{'prefix': '<fim_prefix>', 'middle': '<fim_middle>', 'suffix': '<fim_suffix>'},
                  'ibm-granite/granite-8b-code-base' :{'prefix': '<fim_prefix>', 'middle': '<fim_middle>', 'suffix': '<fim_suffix>'}}


def _sanitize(name: str) -> str:
    """Normalize a model or field name for use in column names."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def _select_scorer(metric: str):
    metric_key = metric.lower()
    if metric_key in {"bertscore", "bert"}:
        return "bertscore", bert_score
    if metric_key in {"bartscore", "bart"}:
        return "bartscore", bart_score
    raise ValueError(f"Unknown metric '{metric}'. Use 'bertscore' or 'bartscore'.")


def compute_score_columns(
    dataset: Dataset,
    field: str,
    scoring_models: Iterable[str],
    metric: str,
    device: str = "cpu",
    device_map: str | None = None,
    torch_dtype=None,
    offload_folder: str | None = None,
    batch_size: int = 16,
) -> Tuple[Dataset, List[Tuple[str, float, float, float]]]:
    """
    Add BERTScore/BARTScore columns for each model in scoring_models and return averages.
    """
    metric_prefix, scorer = _select_scorer(metric)
    results_suffix = field.split("/")[-1]
    results = []

    for scoring_model in scoring_models:
        model_short = _sanitize(scoring_model)
        col_p = f"{metric_prefix}_P_{results_suffix}_{model_short}"
        col_r = f"{metric_prefix}_R_{results_suffix}_{model_short}"
        col_f = f"{metric_prefix}_F1_{results_suffix}_{model_short}"

        def make_compute(
            field=field,
            scoring_model=scoring_model,
            col_p=col_p,
            col_r=col_r,
            col_f=col_f,
        ):
            if metric == "bertscore":
                def compute_bertscore(batch):
                    preds = batch[f"predicted_comment_{field}"]
                    refs = batch["original_comment"]
                    context = batch[f"masked_data_{field}"]
                    prefixes = []
                    suffixes = []
                    for ctx in context:
                        split_ctx = ctx.split(FIM_TOKEN_DICT[field]["suffix"]) 
                        prefixes.append(split_ctx[0].replace(FIM_TOKEN_DICT[field]["prefix"], ""))
                        suffixes.append(split_ctx[1].replace(FIM_TOKEN_DICT[field]["middle"], ""))

                    p_scores, r_scores, f_scores = scorer(
                        preds,
                        refs,
                        model_type=scoring_model,
                        device=device,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        offload_folder=offload_folder,
                        prefixes=prefixes,
                        suffixes=suffixes,
                        verbose=False,
                    )
                    return {
                        col_p: [float(p) for p in p_scores],
                        col_r: [float(r) for r in r_scores],
                        col_f: [float(f) for f in f_scores],
                    }
                return compute_bertscore
            elif metric == "bartscore":
                def compute_bartscore(batch):
                    preds = batch[f"predicted_comment_{field}"]
                    refs = batch["original_comment"]
                    context = batch[f"masked_data_{field}"]
                    prefixes = []
                    suffixes = []
                    for ctx in context:
                        split_ctx = ctx.split(FIM_TOKEN_DICT[field]["suffix"]) 
                        prefixes.append(split_ctx[0].replace(FIM_TOKEN_DICT[field]["prefix"], ""))
                        suffixes.append(split_ctx[1].replace(FIM_TOKEN_DICT[field]["middle"], ""))

                    scores = scorer(
                        preds,
                        refs,
                        model_type=scoring_model,
                        device=device,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        offload_folder=offload_folder,
                        prefixes=prefixes,
                        suffixes=suffixes,
                        # verbose=False,
                    )
                    return {
                        col_p: scores,  # BARTScore does not have separate P/R/F1
                        col_r: scores,
                        col_f: scores,
                    }
                return compute_bartscore

        dataset = dataset.map(make_compute(), batched=True, batch_size=batch_size)

        p_vals = np.array(dataset[col_p])
        r_vals = np.array(dataset[col_r])
        f_vals = np.array(dataset[col_f])
        results.append((model_short, float(p_vals.mean()), float(r_vals.mean()), float(f_vals.mean())))

    return dataset, results


def maybe_login_from_env(token: str | None) -> None:
    """Login to the HF Hub if a token is provided."""
    if token:
        login(token=token)


def run_experiment(
    dataset_id: str,
    subset: str,
    split: str,
    llm_fields: Iterable[str],
    scoring_models: Iterable[str],
    metric: str,
    device: str,
    device_map: str | None,
    torch_dtype,
    offload_folder: str | None,
    batch_size: int,
    save_path: str | None,
) -> Tuple[Dataset, dict]:
    ds = load_dataset(dataset_id, subset)[split]
    all_results = {}

    for field in llm_fields:
        ds, results = compute_score_columns(
            dataset=ds,
            field=field,
            scoring_models=scoring_models,
            metric=metric,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            offload_folder=offload_folder,
            batch_size=batch_size,
        )
        print(f"\nField: {field}")
        for name, p_mean, r_mean, f_mean in results:
            print(f"{name:30s} ({metric}) P: {p_mean:.4f} | R: {r_mean:.4f} | F1: {f_mean:.4f}")
        all_results[field] = results

    if save_path:
        ds.to_json(save_path)
        print(f"\nSaved scored split to {save_path}")

    return ds, all_results

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BERTScore experiments.")
    parser.add_argument(
        "--dataset-id",
        default="AISE-TUDelft/multilingual-code-comments",
        help="HF dataset repo ID.",
    )
    parser.add_argument("--subset", default="English", help="Dataset subset/config name.")
    parser.add_argument("--split", default="train", help="Dataset split to score.")
    parser.add_argument(
        "--llm-fields",
        nargs="+",
        default=["google/codegemma-7b"],
        help="One or more dataset fields containing model generations.",
    )
    parser.add_argument(
        "--scoring-models",
        nargs="+",
        default=[
            "google/embeddinggemma-300m",
            "weiweishi/roc-bert-base-zh",
            "allegro/herbert-large-cased",
            "GroNLP/bert-base-dutch-cased",
        ],
        help="BERTScore model checkpoints to evaluate with.",
    )
    parser.add_argument("--device", default="cpu", help="Device for BERTScore (e.g., cpu or cuda:0).")
    parser.add_argument(
        "--device-map",
        default=None,
        help="Pass 'auto' to shard/offload large models across available devices.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Optional torch dtype for model (e.g., float16, bfloat16) to save memory.",
    )
    parser.add_argument(
        "--offload-folder",
        default=None,
        help="Optional folder for disk offload when using device_map=auto.",
    )
    parser.add_argument(
        "--ref-prefix",
        default="",
        help="Optional prefix context added before reference text (masked out in scoring for transformers backend).",
    )
    parser.add_argument(
        "--ref-suffix",
        default="",
        help="Optional suffix context added after reference text (masked out in scoring for transformers backend).",
    )
    parser.add_argument(
        "--metric",
        default="bertscore",
        choices=["bertscore", "bartscore", "bert", "bart"],
        help="Scoring metric to use.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for dataset.map.")
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional path to save the scored split as JSON.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token (env HF_TOKEN respected).",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    maybe_login_from_env(args.hf_token)
    run_experiment(
        dataset_id=args.dataset_id,
        subset=args.subset,
        split=args.split,
        llm_fields=args.llm_fields,
        scoring_models=args.scoring_models,
        metric=args.metric,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        offload_folder=args.offload_folder,
        batch_size=args.batch_size,
        save_path=args.save_path,
    )
