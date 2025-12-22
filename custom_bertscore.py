"""
Lightweight BERTScore-style implementation using Hugging Face transformers.

Supports optional reference prefix/suffix for context; only the reference span
is used in scoring (context tokens are masked out).

Large-model support:
- device_map="auto" for multi-GPU or CPU/GPU offload
- torch_dtype (e.g., "float16", "bfloat16") to reduce memory
- offload_folder for disk offload when using device_map
"""
from functools import lru_cache
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


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
    return AutoModel.from_pretrained(
        model_name, device_map=device_map, torch_dtype=dtype, offload_folder=offload_folder
    )


def _normalize_context_list(value: str | Iterable[str], name: str, expected_len: int) -> List[str]:
    if isinstance(value, str):
        return [value] * expected_len
    if not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a string or iterable of strings")
    values = list(value)
    if len(values) != expected_len:
        raise ValueError(f"{name} must have exactly {expected_len} entries")
    if not all(isinstance(v, str) for v in values):
        raise ValueError(f"All entries in {name} must be strings")
    return values


def _embed_transformers(
    texts: Iterable[str],
    model_name: str,
    device: str,
    device_map: str | None,
    torch_dtype,
    offload_folder: str | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = _load_tokenizer(model_name)
    model = _load_model(model_name, device_map, torch_dtype, offload_folder)
    # If not sharded, keep model on the requested device
    if device_map is None and str(next(model.parameters()).device) != device:
        model.to(device)

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # Move inputs unless model is already sharded
    if device_map is None:
        encoded = encoded.to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        hidden = outputs.last_hidden_state  # [batch, seq, hidden]

    attn_mask = encoded["attention_mask"]
    if device_map is not None:
        attn_mask = attn_mask.to(hidden.device)

    return hidden


def _embed_transformers_with_context(
    refs: Iterable[str],
    model_name: str,
    device: str,
    device_map: str | None,
    torch_dtype,
    offload_folder: str | None,
    prefixes: List[str],
    suffixes: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = _load_tokenizer(model_name)
    model = _load_model(model_name, device_map, torch_dtype, offload_folder)
    if device_map is None and str(next(model.parameters()).device) != device:
        model.to(device)

    if len(prefixes) != len(refs) or len(suffixes) != len(refs):
        raise ValueError("prefixes and suffixes must match refs length")

    texts = []
    prefix_lens: List[int] = []
    suffix_lens: List[int] = []
    for p, ref, s in zip(prefixes, refs, suffixes):
        texts.append(f"{p}{ref}{s}")
        prefix_lens.append(len(tokenizer(p, add_special_tokens=False)["input_ids"]))
        suffix_lens.append(len(tokenizer(s, add_special_tokens=False)["input_ids"]))
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    if device_map is None:
        encoded = encoded.to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        hidden = outputs.last_hidden_state

    attn_mask = encoded["attention_mask"]
    special_mask = encoded["special_tokens_mask"]
    if device_map is not None:
        attn_mask = attn_mask.to(hidden.device)
        special_mask = special_mask.to(hidden.device)

    keep_token_indices: List[List[int]] = []
    lengths: List[int] = []
    for keep_mask, spec_mask, pre_len, suf_len in zip(attn_mask, special_mask, prefix_lens, suffix_lens):
        text_indices = [
            idx for idx, (keep, special) in enumerate(zip(keep_mask.tolist(), spec_mask.tolist())) if keep == 1 and special == 0
        ]
        start = min(pre_len, len(text_indices))
        end = len(text_indices) - suf_len if suf_len <= len(text_indices) else 0
        end = max(start, end)
        indices = text_indices[start:end]
        keep_token_indices.append(indices)
        lengths.append(len(indices))

    max_len = max(lengths) if lengths else 0
    hidden_size = hidden.size(-1)
    trimmed_hidden = hidden.new_zeros((len(refs), max_len, hidden_size))
    ref_mask = torch.zeros((len(refs), max_len), device=hidden.device, dtype=torch.bool)

    for i, indices in enumerate(keep_token_indices):
        if not indices:
            continue
        idx_tensor = torch.tensor(indices, device=hidden.device)
        seq = hidden[i].index_select(0, idx_tensor)
        trimmed_hidden[i, : seq.size(0)] = seq
        ref_mask[i, : seq.size(0)] = True

    return trimmed_hidden


def score(
    cands: List[str],
    refs: List[str],
    model_type: str,
    device: str = "cpu",
    device_map: str | None = None,
    torch_dtype=None,
    offload_folder: str | None = None,
    prefixes: List[str] = [],
    suffixes: List[str] = [],
    verbose: bool = False,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute a simplified BERTScore for aligned candidate/reference pairs.
    """
    if len(cands) != len(refs):
        raise ValueError("cands and refs must have the same length")

    cand_embeds = _embed_transformers_with_context(
        cands, model_type, device, device_map, torch_dtype, offload_folder, prefixes, suffixes
    )
    ref_embeds = _embed_transformers_with_context(
        refs, model_type, device, device_map, torch_dtype, offload_folder, prefixes, suffixes
    )

    # Decide where to run math
    target_device = cand_embeds.device if device_map else device
    cand_embeds = cand_embeds.to(target_device)
    ref_embeds = ref_embeds.to(target_device)

    cand_embeds = F.normalize(cand_embeds, p=2, dim=-1)
    ref_embeds = F.normalize(ref_embeds, p=2, dim=-1)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for i in range(len(cands)):
        c_tokens = cand_embeds[i]
        r_tokens = ref_embeds[i]

        if c_tokens.numel() == 0 or r_tokens.numel() == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue

        sim = torch.matmul(c_tokens, r_tokens.transpose(0, 1))

        p = sim.max(dim=1).values.mean().item()
        r = sim.max(dim=0).values.mean().item()
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return precisions, recalls, f1s
