import os
import pathlib
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoTokenizer, AutoConfig

from .utils import (bert_cos_score_idf, cache_scibert, get_bert_embedding,
                    get_hash, get_idf_dict, get_model, get_tokenizer,
                    lang2model, model2layers, sent_encode)

__all__ = ["score", "plot_example"]

def truncate(prefix: str, text: str, suffix: str, max_len: int, tokenizer) -> str:
    max_len = max_len - 3  # account for special tokens
    pre_tokens = tokenizer.tokenize(prefix)
    suf_tokens = tokenizer.tokenize(suffix)
    text_tokens = tokenizer.tokenize(text)
    token_budget = max_len - len(text_tokens)

    # Need to truncate text itself
    if token_budget <= 0:
        # No budget for prefix/suffix, truncate text only
        text_tokens = text_tokens[:max_len]
        return tokenizer.convert_tokens_to_string(text_tokens)

    if len(pre_tokens) + len(suf_tokens) + len(text_tokens) < max_len:
        return tokenizer.convert_tokens_to_string(pre_tokens + text_tokens + suf_tokens)
    # Do we need to truncate both sides
    half_budget = token_budget // 2
    n_pre_trunc = min(len(pre_tokens), half_budget)
    n_suf_trunc = min(len(suf_tokens), half_budget)
    
    #if we dont need to truncate prefix, add remaining budget to suffix
    if n_pre_trunc < half_budget:
        n_suf_trunc += half_budget - n_pre_trunc
    #if we dont need to truncate suffix, add remaining budget to prefix
    if n_suf_trunc < half_budget:
        n_pre_trunc += half_budget - n_suf_trunc

    tokens_pre_trunc = pre_tokens[-n_pre_trunc:]
    tokens_suf_trunc = suf_tokens[:n_suf_trunc]
    return tokenizer.convert_tokens_to_string(tokens_pre_trunc + text_tokens + tokens_suf_trunc)

def score(
    cands,
    refs,
    model_type=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    lang=None,
    return_hash=False,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
    strip_prefix="",
    strip_suffix="",
    strip_prefix_ref=None,
    strip_suffix_ref=None,
    strip_prefix_cand=None,
    strip_suffix_cand=None,
):
    """
    BERTScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `baseline_path` (str): customized baseline file
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        - :param: `strip_prefix` (str): optional prefix to add for context then ignore in similarity
        - :param: `strip_suffix` (str): optional suffix to add for context then ignore in similarity
        - :param: `strip_prefix_ref` (str or list): optional prefix for references only; list length must match refs
        - :param: `strip_suffix_ref` (str or list): optional suffix for references only; list length must match refs
        - :param: `strip_prefix_cand` (str or list): optional prefix for candidates only; list length must match cands
        - :param: `strip_suffix_cand` (str or list): optional suffix for candidates only; list length must match cands

    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have
                  multiple references, the returned score of this candidate is
                  the *best* score among all references.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"

    assert (
        lang is not None or model_type is not None
    ), "Either lang or model_type should be specified"

    def _broadcast_affix(shared, per_item, length):
        if isinstance(per_item, list):
            assert len(per_item) == length, "Per-item affix list length must match inputs"
            return per_item
        return [shared] * length if per_item is None else [per_item] * length

    # Broadcast affixes for cands and refs (used for context during embedding)
    cand_prefixes = _broadcast_affix(strip_prefix, strip_prefix_cand, len(cands))
    cand_suffixes = _broadcast_affix(strip_suffix, strip_suffix_cand, len(cands))

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        if model_type in model2layers:
            num_layers = model2layers[model_type]
        else:
            num_layers = int(AutoConfig.from_pretrained(model_type).num_hidden_layers * 0.75)

    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    model = get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def _count_tokens(text):
        return len(tokenizer.tokenize(text))

    max_seq_length = AutoConfig.from_pretrained(model_type).max_position_embeddings

    embed_cands = []
    embed_refs = []
    cand_trim_heads = []
    cand_trim_tails = []
    ref_trim_heads = []
    ref_trim_tails = []
    ref_group_boundaries = None

    # Fast path: each candidate has exactly one reference (all refs are strings)
    if isinstance(refs[0], str):
        prefixes = _broadcast_affix(strip_prefix, strip_prefix_ref, len(refs))
        suffixes = _broadcast_affix(strip_suffix, strip_suffix_ref, len(refs))

        for cand, ref_text, cand_pre, cand_suf, rpre, rsuf in zip(
            cands, refs, cand_prefixes, cand_suffixes, prefixes, suffixes
        ):
            affixed_ref = truncate(rpre, ref_text, rsuf, max_seq_length, tokenizer)
            affixed_cand = truncate(cand_pre, cand, cand_suf, max_seq_length, tokenizer)

            embed_cands.append(affixed_cand)
            embed_refs.append(affixed_ref)
            cand_trim_heads.append(_count_tokens(cand_pre))
            cand_trim_tails.append(_count_tokens(cand_suf))
            ref_trim_heads.append(_count_tokens(rpre))
            ref_trim_tails.append(_count_tokens(rsuf))
    else:
        # refs is a list of reference lists
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0

        # Prepare per-group affixes: accept nested list matching refs structure
        if isinstance(strip_prefix_ref, list) and strip_prefix_ref and isinstance(strip_prefix_ref[0], list):
            assert len(strip_prefix_ref) == len(ori_refs), "strip_prefix_ref groups must align with refs groups"
            ref_prefix_groups = strip_prefix_ref
        else:
            ref_prefix_groups = [strip_prefix_ref] * len(ori_refs)

        if isinstance(strip_suffix_ref, list) and strip_suffix_ref and isinstance(strip_suffix_ref[0], list):
            assert len(strip_suffix_ref) == len(ori_refs), "strip_suffix_ref groups must align with refs groups"
            ref_suffix_groups = strip_suffix_ref
        else:
            ref_suffix_groups = [strip_suffix_ref] * len(ori_refs)

        for cand, ref_group, pre_group, suf_group, cand_pre, cand_suf in zip(
            ori_cands, ori_refs, ref_prefix_groups, ref_suffix_groups, cand_prefixes, cand_suffixes
        ):
            prefixes = _broadcast_affix(strip_prefix, pre_group, len(ref_group))
            suffixes = _broadcast_affix(strip_suffix, suf_group, len(ref_group))
            cand_pre_tokens = _count_tokens(cand_pre)
            cand_suf_tokens = _count_tokens(cand_suf)
            affixed_cand = truncate(cand_pre, cand, cand_suf, max_seq_length, tokenizer)
            for ref_text, rpre, rsuf in zip(ref_group, prefixes, suffixes):
                affixed_ref = truncate(rpre, ref_text, rsuf, max_seq_length, tokenizer)
                embed_refs.append(affixed_ref)
                ref_trim_heads.append(_count_tokens(rpre))
                ref_trim_tails.append(_count_tokens(rsuf))

                embed_cands.append(affixed_cand)
                cand_trim_heads.append(cand_pre_tokens)
                cand_trim_tails.append(cand_suf_tokens)
                cands.append(cand)
                refs.append(ref_text)
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)
        embed_cands = [_clamp_text(t) for t in embed_cands]
        embed_refs = [_clamp_text(t) for t in embed_refs]
    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        if verbose:
            print("using predefined IDF dict...")
        idf_dict = idf
    else:
        if verbose:
            print("preparing IDF dict...")
        start = time.perf_counter()
        # IDF should reflect the affixed reference text
        idf_dict = get_idf_dict(embed_refs, tokenizer, nthreads=nthreads)
        if verbose:
            print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(
        model,
        embed_refs,
        embed_cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
        ref_trim_heads=ref_trim_heads,
        ref_trim_tails=ref_trim_tails,
        hyp_trim_heads=cand_trim_heads,
        hyp_trim_tails=cand_trim_tails,
    ).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    use_custom_baseline = baseline_path is not None
    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(
                os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv"
            )
        if os.path.isfile(baseline_path):
            if not all_layers:
                baselines = torch.from_numpy(
                    pd.read_csv(baseline_path).iloc[num_layers].to_numpy()
                )[1:].float()
            else:
                baselines = (
                    torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:]
                    .unsqueeze(1)
                    .float()
                )

            all_preds = (all_preds - baselines) / (1 - baselines)
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}",
                file=sys.stderr,
            )

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(
            f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec"
        )

    if return_hash:
        return tuple(
            [
                out,
                get_hash(
                    model_type,
                    num_layers,
                    idf,
                    rescale_with_baseline,
                    use_custom_baseline=use_custom_baseline,
                    use_fast_tokenizer=use_fast_tokenizer,
                ),
            ]
        )

    return out


def plot_example(
    candidate,
    reference,
    model_type=None,
    num_layers=None,
    lang=None,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
    fname="",
):
    """
    BERTScore metric.

    Args:
        - :param: `candidate` (str): a candidate sentence
        - :param: `reference` (str): a reference sentence
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        - :param: `fname` (str): path to save the output plot
    """
    assert isinstance(candidate, str)
    assert isinstance(reference, str)

    assert (
        lang is not None or model_type is not None
    ), "Either lang or model_type should be specified"

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]

    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    model = get_model(model_type, num_layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    idf_dict = defaultdict(lambda: 1.0)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0

    hyp_embedding, masks, padded_idf = get_bert_embedding(
        [candidate], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    ref_embedding, masks, padded_idf = get_bert_embedding(
        [reference], model, tokenizer, idf_dict, device=device, all_layers=False
    )
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    sim = sim.squeeze(0).cpu()

    # remove [CLS] and [SEP] tokens
    r_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, reference)][1:-1]
    h_tokens = [tokenizer.decode([i]) for i in sent_encode(tokenizer, candidate)][1:-1]
    sim = sim[1:-1, 1:-1]

    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(
                os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv"
            )
        if os.path.isfile(baseline_path):
            baselines = torch.from_numpy(
                pd.read_csv(baseline_path).iloc[num_layers].to_numpy()
            )[1:].float()
            sim = (sim - baselines[2].item()) / (1 - baselines[2].item())
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}",
                file=sys.stderr,
            )

    fig, ax = plt.subplots(figsize=(len(r_tokens), len(h_tokens)))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(r_tokens)))
    ax.set_yticks(np.arange(len(h_tokens)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(r_tokens, fontsize=10)
    ax.set_yticklabels(h_tokens, fontsize=10)
    ax.grid(False)
    plt.xlabel("Reference (tokenized)", fontsize=14)
    plt.ylabel("Candidate (tokenized)", fontsize=14)
    title = "Similarity Matrix"
    if rescale_with_baseline:
        title += " (after Rescaling)"
    plt.title(title, fontsize=14)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)
    fig.colorbar(im, cax=cax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(h_tokens)):
        for j in range(len(r_tokens)):
            text = ax.text(
                j,
                i,
                "{:.3f}".format(sim[i, j].item()),
                ha="center",
                va="center",
                color="k" if sim[i, j].item() < 0.5 else "w",
            )

    fig.tight_layout()
    if fname != "":
        plt.savefig(fname, dpi=100)
        print("Saved figure to file: ", fname)
    plt.show()
