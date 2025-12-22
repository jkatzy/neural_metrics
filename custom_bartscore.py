"""
Lightweight BARTScore-style implementation using Hugging Face seq2seq models.

Computes average token log-probabilities of a target string conditioned on a
source string (defaults to candidate | reference). The model checkpoint is
fully configurable, so you can swap in any compatible seq2seq model.
"""
from functools import lru_cache
from typing import Iterable, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _resolve_dtype(torch_dtype):
    if torch_dtype is None:
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str):
        if not hasattr(torch, torch_dtype):
            raise ValueError(f"Unknown torch dtype string: {torch_dtype}")
        return getattr(torch, torch_dtype)
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=2)
def _load_model(model_name: str, device_map: str | None, torch_dtype, offload_folder: str | None):
    dtype = _resolve_dtype(torch_dtype)
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=dtype, offload_folder=offload_folder
    )

def _encode_batch(
    tokenizer,
    sources: Iterable[str],
    targets: Iterable[str],
    device: str,
    device_map: str | None,
    max_source_length: int | None,
    max_target_length: int | None,
    prefixes: List[str],
    suffixes: List[str],
):
    source_texts = [f"{p}{s}{suf}" for p, s, suf in zip(prefixes, sources, suffixes)]
    target_texts = [f"{p}{t}{suf}" for p, t, suf in zip(prefixes, targets, suffixes)]

    model_inputs = tokenizer(
        source_texts,
        padding=True,
        truncation=True,
        max_length=max_source_length,
        return_tensors="pt",
    )
    target_tokens = tokenizer(
        text_target=target_texts,
        padding=True,
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    target_ids = target_tokens["input_ids"]
    attention_mask = target_tokens["attention_mask"].bool()
    special_mask = target_tokens["special_tokens_mask"].bool()

    prefix_token_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prefixes]
    suffix_token_lens = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in suffixes]

    target_only_masks = []
    for attn, spec, pre_len, suf_len in zip(attention_mask, special_mask, prefix_token_lens, suffix_token_lens):
        text_indices = [idx for idx, (keep, sp) in enumerate(zip(attn.tolist(), spec.tolist())) if keep and sp == 0]
        start = min(pre_len, len(text_indices))
        end = len(text_indices) - suf_len if suf_len <= len(text_indices) else 0
        end = max(start, end)
        keep_indices = text_indices[start:end]
        mask = torch.zeros_like(attn, dtype=torch.bool)
        if keep_indices:
            mask[torch.tensor(keep_indices, device=mask.device)] = True
        target_only_masks.append(mask)

    target_mask = torch.stack(target_only_masks, dim=0)
    labels_for_loss = target_ids.masked_fill(~target_mask, -100)

    if device_map is None:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        target_ids = target_ids.to(device)
        target_mask = target_mask.to(device)
        labels_for_loss = labels_for_loss.to(device)

    return model_inputs, labels_for_loss, target_ids, target_mask


def score(
    cands: List[str],
    refs: List[str],
    model_type: str,
    device: str = "cpu",
    device_map: str | None = None,
    torch_dtype=None,
    offload_folder: str | None = None,
    batch_size: int = 4,
    max_source_length: int | None = None,
    max_target_length: int | None = None,
    ref_as_source: bool = True,
    normalize: bool = True,
    prefixes: List[str] = [],
    suffixes: List[str] = [],
) -> List[float]:
    """
    Compute BARTScore-style log probabilities for aligned candidate/reference pairs.

    Args:
        cands: Candidate strings to score.
        refs: Reference strings for conditioning.
        model_type: HF model name or path for any seq2seq checkpoint (e.g., BART/T5/Flan).
        device: Device used when not sharding (e.g., "cpu" or "cuda:0").
        device_map: Pass "auto" to shard/offload large models.
        torch_dtype: Optional dtype (e.g., "float16", torch.bfloat16) when loading the model.
        offload_folder: Optional folder used when sharding/offloading.
        batch_size: Number of examples per forward pass.
        max_source_length: Optional max length for the conditioning text.
        max_target_length: Optional max length for the scored text.
        ref_as_source: If True (default), score p(candidate | reference); otherwise p(reference | candidate).
        normalize: When True, return mean log-prob per target token; otherwise return the sum of log-probs.
        source_prefix/source_suffix: Optional context added around the conditioning text; also used for targets.

    Returns:
        List of log-probability scores (one per pair); higher is better.
    """
    if len(cands) != len(refs):
        raise ValueError("cands and refs must have the same length")

    tokenizer = _load_tokenizer(model_type)
    model = _load_model(model_type, device_map, torch_dtype, offload_folder)
    try:
        first_param_device = next(model.parameters()).device
    except StopIteration:
        first_param_device = torch.device("cpu")

    if device_map is None and str(first_param_device) != device:
        model.to(device)

    sources = refs if ref_as_source else cands
    targets = cands if ref_as_source else refs
    scores: List[float] = []

    for start in range(0, len(cands), batch_size):
        end = start + batch_size
        batch_sources = sources[start:end]
        batch_targets = targets[start:end]
        batch_prefixes = prefixes[start:end]
        batch_suffixes = suffixes[start:end]

        inputs, labels, target_ids, target_mask = _encode_batch(
            tokenizer=tokenizer,
            sources=batch_sources,
            targets=batch_targets,
            device=device,
            device_map=device_map,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            prefixes=batch_prefixes,
            suffixes=batch_suffixes,
        )

        with torch.no_grad():
            outputs = model(**inputs, labels=labels) #TODO: Ensure that the labels are only calculating the loss for the target part of the sequence

        log_probs = outputs.logits.log_softmax(dim=-1)
        target_ids = target_ids.to(log_probs.device)
        target_mask = target_mask.to(log_probs.device)

        token_logps = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps.masked_fill(~target_mask, 0.0)

        lengths = target_mask.sum(dim=1).clamp(min=1)
        summed = token_logps.sum(dim=1)
        batch_scores = summed / lengths if normalize else summed

        scores.extend(float(s) for s in batch_scores.cpu())

    return scores
