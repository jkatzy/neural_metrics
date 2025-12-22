# Generation Scoring Project

A Python project for scoring text generations against ground truths using multiple evaluation metrics and HuggingFace datasets.

## Features

- **Multiple Metrics**: BLEU, Exact Match, Token Overlap, and ROUGE-L
- **HuggingFace Integration**: Uses the datasets library with efficient `map()` function
- **Batch Processing**: Support for scoring multiple generations at once
- **Flexible Dataset Handling**: Create datasets from dictionaries, JSON files, or existing HuggingFace datasets

## Project Structure

```
├── metrics.py              # Core metrics implementations
├── scorer.py              # Main scoring class
├── dataset_utils.py       # Dataset loading and processing utilities
├── example_data.json      # Sample data for testing
└── notebook_drivers.ipynb # Jupyter notebook with examples and drivers
```

## Installation

Install required dependencies from the pinned list:

```bash
pip install -r requirements.txt
```

To use the project notebook with its own Jupyter kernel:

```bash
python -m ipykernel install --user --name babel --display-name "Babel"
```

## Usage

### Basic Usage

```python
from scorer import GenerationScorer

# Initialize scorer with specific metrics
scorer = GenerationScorer(metrics=["bleu", "exact_match", "token_overlap", "rouge_l"])

# Score a single generation
scores = scorer.score_single(
    generated="the quick brown fox",
    ground_truth="the quick brown fox jumps"
)
print(scores)  # {'bleu': 0.8, 'exact_match': 0.0, 'token_overlap': 0.667, 'rouge_l': 0.857}
```

### Using with HuggingFace Datasets

```python
from datasets import Dataset
from scorer import GenerationScorer

# Create or load a dataset
data = {
    "id": [1, 2, 3],
    "generation": ["hello world", "goodbye world", "test"],
    "ground_truth": ["hello world", "hello world", "test"]
}
dataset = Dataset.from_dict(data)

# Score using the map function
scorer = GenerationScorer()
scored_dataset = dataset.map(scorer.score_dataset_map_function)

print(scored_dataset)
```

## Available Metrics

- **BLEU**: Token overlap with reference based on word frequency
- **Exact Match**: 1.0 if generated equals ground truth, 0.0 otherwise
- **Token Overlap**: Jaccard similarity between generated and ground truth tokens
- **ROUGE-L**: F1 score based on longest common subsequence of words

## License

MIT
