"""
Dataset utilities for loading and processing generations and ground truths.
"""
from typing import List, Dict, Any, Callable
from datasets import Dataset, load_dataset
import json


def create_dataset_from_dict(
    data: List[Dict[str, str]],
    generation_key: str = "generation",
    ground_truth_key: str = "ground_truth"
) -> Dataset:
    """
    Create a HuggingFace Dataset from a list of dictionaries.
    
    Args:
        data: List of dictionaries with generation and ground truth pairs
        generation_key: Key in dict for generated text
        ground_truth_key: Key in dict for ground truth text
        
    Returns:
        HuggingFace Dataset object
    """
    return Dataset.from_dict({
        "id": list(range(len(data))),
        generation_key: [item[generation_key] for item in data],
        ground_truth_key: [item[ground_truth_key] for item in data]
    })


def create_dataset_from_json(
    json_path: str,
    generation_key: str = "generation",
    ground_truth_key: str = "ground_truth"
) -> Dataset:
    """
    Create a HuggingFace Dataset from a JSON file.
    
    Args:
        json_path: Path to JSON file
        generation_key: Key in dict for generated text
        ground_truth_key: Key in dict for ground truth text
        
    Returns:
        HuggingFace Dataset object
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return create_dataset_from_dict(data, generation_key, ground_truth_key)


def load_dataset_from_hub(
    dataset_name: str,
    split: str = "train",
    generation_key: str = "generation",
    ground_truth_key: str = "ground_truth"
) -> Dataset:
    """
    Load a dataset that already lives on the HuggingFace Hub.

    Args:
        dataset_name: Repository ID on the Hub (e.g. "username/dataset")
        split: Which split to download (e.g. "train", "validation")
        generation_key: Column name that contains generated text
        ground_truth_key: Column name that contains reference text

    Returns:
        HuggingFace Dataset with standard generation/ground_truth columns
    """
    dataset = load_dataset(dataset_name, split=split)

    missing = [
        column_name
        for column_name in (generation_key, ground_truth_key)
        if column_name not in dataset.column_names
    ]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required columns: {missing}"
        )

    rename_map = {}
    if generation_key != "generation":
        rename_map[generation_key] = "generation"
    if ground_truth_key != "ground_truth":
        rename_map[ground_truth_key] = "ground_truth"

    if rename_map:
        dataset = dataset.rename_columns(rename_map)

    return dataset


def apply_metrics_to_dataset(
    dataset: Dataset,
    scoring_func: Callable[[Dict[str, str]], Dict[str, Any]],
    input_columns: List[str] = None,
    batched: bool = False,
    batch_size: int = 32
) -> Dataset:
    """
    Apply a scoring function to a dataset using the map function.
    
    Args:
        dataset: HuggingFace Dataset to score
        scoring_func: Function that takes a dict and returns scores
        input_columns: Columns to keep (others will be removed)
        batched: Whether to apply function in batches
        batch_size: Batch size for processing
        
    Returns:
        Dataset with added score columns
    """
    if input_columns is None:
        input_columns = list(dataset.column_names)
    
    # Apply the scoring function using map
    scored_dataset = dataset.map(
        scoring_func,
        batched=batched,
        batch_size=batch_size if batched else None,
        remove_columns=[col for col in dataset.column_names if col not in input_columns]
    )
    
    return scored_dataset


def save_dataset(dataset: Dataset, output_path: str) -> None:
    """
    Save dataset to a JSON file.
    
    Args:
        dataset: HuggingFace Dataset to save
        output_path: Path to save the dataset
    """
    dataset.to_json(output_path)
