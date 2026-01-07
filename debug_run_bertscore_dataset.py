#!/usr/bin/env python
"""
Quick debug harness to exercise run_bertscore_dataset.main on a tiny in-memory
dataset. Adjust the sample strings below as needed, then run:

    python debug_run_bertscore_dataset.py
"""
from datasets import Dataset

import bert_score


def build_dummy_data():
    refs = ["cat", "bat"]
    cands = ["mat", "pat"]
    # Example affixes (could be empty strings)
    cand_prefixes = ["[REF] kowefiowqjfeiouqwfiouweqhfiohwqfoiuwhqfoiuhw    iofh    pqmiowm oenfpufpwiqjipeq ", "[REF]  epo pockwecmqwpiunvoiurqnpwi;owqmclsdkjnciueoiueqn"]
    cand_suffixes = [" ef   wko'woiejfwqpoijoiuwehiof[/REF]", " eqwfoijcimlksancpowiejcuoiwqcoiwucqcijnco;;clkwopcjiqpw[/REF]"]
    ref_prefixes = ["wqew   ecwecwc wecqevvsdverberb[REF] ", "evaoijweopjcpowijclajeclk[REF] "]
    ref_suffixes = ["waecpawkmcopiawjecpoijepciaince [/REF]", " wpockawpoijcpoawjicpoajepocjsa;kljclksdjcopisecajscoasijecji[/REF]"]
    return cands[0:1], refs[0:1], cand_prefixes[0:1], cand_suffixes[0:1], ref_prefixes[0:1], ref_suffixes[0:1]


def main():
    cands, refs, cand_prefixes, cand_suffixes, ref_prefixes, ref_suffixes = build_dummy_data()
    P, R, F = bert_score.score(
        cands,
        refs,
        model_type="roberta-large",
        lang="en",
        strip_prefix_cand=None,
        strip_suffix_cand=None,
        strip_prefix_ref=None,
        strip_suffix_ref=None,
    )
    print("P:", P.tolist())
    print("R:", R.tolist())
    print("F:", F.tolist())
    P, R, F = bert_score.score(
        cands,
        refs,
        model_type="roberta-large",
        lang="en",
        strip_prefix_cand=cand_prefixes,
        strip_suffix_cand=cand_suffixes,
        strip_prefix_ref=ref_prefixes,
        strip_suffix_ref=ref_suffixes,
    )
    print("P:", P.tolist())
    print("R:", R.tolist())
    print("F:", F.tolist())

if __name__ == "__main__":
    main()
