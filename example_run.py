"""
Minimal example for running the BERTScore experiment logic with CLI arguments.
"""
import argparse
import os

from experiment_runner import run_experiment, maybe_login_from_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Example runner for BERTScore experiments.")
    parser.add_argument("--dataset-id", default="AISE-TUDelft/multilingual-code-comments")
    parser.add_argument("--subset", default="English")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--llm-fields",
        nargs="+",
        default=["predict_google/codegemma-7b"],
        help="Dataset fields containing model generations.",
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
        help="BERTScore models to use.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-path", default=None, help="Optional path to save scored split as JSON.")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token; environment variable HF_TOKEN is used by default.",
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
        device=args.device,
        batch_size=args.batch_size,
        save_path=args.save_path,
    )
