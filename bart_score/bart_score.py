# %%
import torch
import torch.nn as nn
import traceback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartForConditionalGeneration
from typing import List
import numpy as np
from bert_score.score import truncate
from bert_score.utils import get_tokenizer

def get_model(checkpoint):
    if 'salesforce' in checkpoint.lower():
        if 'codet5p' in checkpoint.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, trust_remote_code=True, use_safetensors=False)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, use_safetensors=False)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return model

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = get_tokenizer(checkpoint, use_fast=True)
        self.model = get_model(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self,
                srcs,
                tgts,
                prefixes_ref=None,
                suffixes_ref=None,
                prefixes_cand=None,
                suffixes_cand=None,
                batch_size=4,
                use_context=False,
            ):
        """ Score a batch of examples """
        score_list = []
        truncated_srcs = []
        truncated_pres = []
        truncated_sufs = []
        for src, pre, suf in zip(srcs, prefixes_cand, suffixes_cand):
            (truncated_pre, truncated_src, truncated_suf) = truncate(
                pre,
                src,
                suf,
                max_len=self.max_length,
                tokenizer=self.tokenizer,
                return_separate=True
            )
            truncated_srcs.append(truncated_src)
            truncated_pres.append(truncated_pre)
            truncated_sufs.append(truncated_suf)
        srcs = truncated_srcs
        prefixes_cand = truncated_pres
        suffixes_cand = truncated_sufs

        truncated_tgts = []
        truncated_pre_refs = []
        truncated_suf_refs = []
        for tgt, pre, suf in zip(tgts, prefixes_ref, suffixes_ref):
            truncated_pre, truncated_tgt, truncated_suf = truncate(
                pre,
                tgt,
                suf,
                max_len=self.max_length,
                tokenizer=self.tokenizer,
                return_separate=True
            )
            truncated_tgts.append(truncated_tgt)
            truncated_pre_refs.append(truncated_pre)
            truncated_suf_refs.append(truncated_suf)
        tgts = truncated_tgts
        prefixes_ref = truncated_pre_refs
        suffixes_ref = truncated_suf_refs

        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            prefix_cand_list = prefixes_cand[i: i + batch_size] if prefixes_cand is not None else ["" for _ in src_list]
            suffix_cand_list = suffixes_cand[i: i + batch_size] if suffixes_cand is not None else ["" for _ in src_list]

            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )

                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )

                    prefix_tokens = self.tokenizer(prefix_cand_list, add_special_tokens=False)['input_ids']
                    suffix_tokens = self.tokenizer(suffix_cand_list, add_special_tokens=False)['input_ids']

                    select_mask = torch.ones_like(tgt_tokens).bool()
                    select_mask[:,1:len(prefix_tokens)+1] = False
                    select_mask[:,-len(suffix_tokens)-1:-1] = False

                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    # logits = logits[select_mask.view(-1)]
                    # tgt_tokens = tgt_tokens[select_mask]
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss[select_mask.view(-1)].view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))
