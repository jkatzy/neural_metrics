"""
GPTScore-style log-probability metric using OpenAI-compatible completion models.

Scores are computed as the average token log-probability of the target text
conditioned on the source text. By default this is p(candidate | reference);
set ref_as_source=False to score p(reference | candidate).

Models: any completion model that supports `echo=True` and `logprobs`
parameters (e.g., gpt-3.5-turbo-instruct, gpt-4o-mini-translate, or compatible
OpenRouter endpoints). Swap models by passing the desired `model_type`.
"""
from functools import lru_cache
from typing import List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Defer error until runtime to keep optional dependency


def _ensure_openai_available():
    if OpenAI is None:
        raise ImportError(
            "The openai package is required for custom_gptscore. Install with `pip install openai>=1.30.0`."
        )


@lru_cache(maxsize=2)
def _get_client(api_key: str | None = None, base_url: str | None = None):
    _ensure_openai_available()
    return OpenAI(api_key=api_key, base_url=base_url)


def _format_text(text: str, prefix: str, suffix: str) -> str:
    return f"{prefix}{text}{suffix}"


def _score_pair(
    client,
    model: str,
    source_text: str,
    target_text: str,
    separator: str,
    normalize: bool,
) -> float:
    prompt = f"{source_text}{separator}{target_text}"
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=0,
        echo=True,
        logprobs=1,
    )

    logprobs = response.choices[0].logprobs
    token_logps = logprobs.token_logprobs or []
    offsets = logprobs.text_offset or [None] * len(token_logps)

    target_start = len(source_text + separator)
    target_token_logps = [
        lp
        for lp, offset in zip(token_logps, offsets)
        if lp is not None and offset is not None and offset >= target_start
    ]

    if not target_token_logps:
        return 0.0

    total = sum(target_token_logps)
    return total / len(target_token_logps) if normalize else total


def score(
    cands: List[str],
    refs: List[str],
    model_type: str,
    api_key: str | None = None,
    base_url: str | None = None,
    separator: str = "\n\n",
    ref_as_source: bool = True,
    normalize: bool = True,
    source_prefix: str = "",
    source_suffix: str = "",
    target_prefix: str = "",
    target_suffix: str = "",
) -> List[float]:
    """
    Compute GPTScore-style log probabilities for aligned candidate/reference pairs.

    Args:
        cands: Candidate strings to score.
        refs: Reference strings for conditioning.
        model_type: Completion model name (must support echo+logprobs).
        api_key: Optional API key; falls back to environment variables if omitted.
        base_url: Optional custom base URL for OpenAI-compatible providers (e.g., OpenRouter).
        separator: String placed between source and target before scoring.
        ref_as_source: If True, compute p(candidate | reference); otherwise p(reference | candidate).
        normalize: When True, return mean log-prob per target token; otherwise return summed log-prob.
        source_prefix/source_suffix: Optional context wrapped around the conditioning text.
        target_prefix/target_suffix: Optional context wrapped around the scored text.

    Returns:
        List of log-probability scores (one per pair); higher is better.
    """
    if len(cands) != len(refs):
        raise ValueError("cands and refs must have the same length")

    client = _get_client(api_key=api_key, base_url=base_url)
    scores: List[float] = []

    for cand, ref in zip(cands, refs):
        source_raw = ref if ref_as_source else cand
        target_raw = cand if ref_as_source else ref

        source_text = _format_text(source_raw, source_prefix, source_suffix)
        target_text = _format_text(target_raw, target_prefix, target_suffix)

        scores.append(
            _score_pair(
                client=client,
                model=model_type,
                source_text=source_text,
                target_text=target_text,
                separator=separator,
                normalize=normalize,
            )
        )

    return scores
