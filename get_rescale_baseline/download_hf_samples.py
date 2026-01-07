#!/usr/bin/env python
import argparse
import os
from typing import Iterable, Tuple

from datasets import Dataset, load_dataset


def infer_text_field(example: dict, preferred: str = "text") -> str:
    if preferred and preferred in example:
        return preferred
    for k, v in example.items():
        if isinstance(v, str):
            return k
    raise ValueError("Unable to infer a text field from the first example")


def stream_samples(
    dataset_id: str, config: str, split: str, text_field: str, num_samples: int
) -> Tuple[str, Iterable[str]]:
    ds = load_dataset(dataset_id, config, split=split, streaming=True)
    iterator = iter(ds)
    first = next(iterator)
    field = text_field or infer_text_field(first)
    yield str(first[field])
    count = 1
    for ex in iterator:
        if count >= num_samples:
            break
        yield str(ex[field])
        count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Stream a HF dataset and materialize N samples locally."
    )
    parser.add_argument("--hf-dataset", required=True, help="HF dataset id (e.g., uonlp/CulturaX)")
    parser.add_argument("--lang", required=True, help="HF config/language to load")
    parser.add_argument("--split", default="train", help="Dataset split to stream (default: train)")
    parser.add_argument(
        "--text-field",
        default=None,
        help="Optional text field name. If omitted, inferred from first example.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to save (default: 10,000).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the saved dataset (default: local_datasets/<dataset>/<lang>).",
    )
    args = parser.parse_args()

    save_root = args.output_dir
    if save_root is None:
        safe_ds = args.hf_dataset.replace("/", "_")
        save_root = os.path.join("local_datasets", safe_ds, args.lang)
    os.makedirs(save_root, exist_ok=True)

    samples = list(
        stream_samples(
            dataset_id=args.hf_dataset,
            config=args.lang,
            split=args.split,
            text_field=args.text_field,
            num_samples=args.num_samples,
        )
    )
    if not samples:
        raise SystemExit("No samples collected; dataset may be empty.")

    text_field = args.text_field or "text"
    Dataset.from_dict({text_field: samples}).save_to_disk(save_root)
    print(f"Saved {len(samples)} samples to {save_root}")


if __name__ == "__main__":
    main()
