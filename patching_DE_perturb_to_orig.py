import os
import re
import json
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, AutoTokenizer

class ResidualTracer:
    def __init__(self, model_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.tok    = AutoTokenizer.from_pretrained(model_name)
        self.yes_id = self._get_single_token_id("yes")
        self.no_id  = self._get_single_token_id("no")
        self.cache, self.hooks = {}, []
        
    def _get_single_token_id(self, text):
        for variant in [f" {text}", text]:
            ids = self.tok(variant, add_special_tokens=False).input_ids
            print("ids: ", ids)
            if len(ids) == 1:
                return ids[0]
        raise ValueError(f"Token '{text}' could not be matched to a single token. Got variants: {self.tok(text, add_special_tokens=False).input_ids}")

    def _get_logits(self, prompt):
        inputs = self.tok(prompt, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        return logits[self.yes_id].item(), logits[self.no_id].item()

    def _num_layers(self):
        cfg = self.model.config
        return getattr(cfg, 'n_layer', cfg.num_hidden_layers)

    def _get_block(self, layer):
        m = self.model
        if hasattr(m, 'transformer'):
            return m.transformer.h[layer]
        if hasattr(m, 'model') and hasattr(m.model, 'layers'):
            return m.model.layers[layer]
        if hasattr(m, 'gpt_neox'):
            return m.gpt_neox.layers[layer]
        raise RuntimeError("Unknown block type")

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def record_residual(self, prompt, tag):
        self.clear_hooks()
        layers = self._num_layers()
        for L in range(layers):
            block = self._get_block(L)
            handle = block.register_forward_pre_hook(
                lambda mod, inp, L=L, tag=tag: self.cache.setdefault((tag,'resid'), {})
                                                 .__setitem__(L, inp[0].detach().clone())
            )
            self.hooks.append(handle)

        with torch.no_grad():
            _ = self.model(**self.tok(prompt, return_tensors='pt'))
        self.clear_hooks()

    def run_residual_layer_level(self, K_index, orig_prompt, pert_prompt, gold_label):
        self.record_residual(orig_prompt, 'orig')
        self.record_residual(pert_prompt, 'pert')

        yes0, no0 = self._get_logits(orig_prompt)
        p_yes0, p_no0 = np.exp(yes0), np.exp(no0)
        base_score = (p_yes0 if gold_label=='yes' else p_no0) / (p_yes0 + p_no0)

        yes_p0, no_p0 = self._get_logits(pert_prompt)
        p_yes_p0, p_no_p0 = np.exp(yes_p0), np.exp(no_p0)
        base_score_p = (p_yes_p0 if gold_label=='yes' else p_no_p0) / (p_yes_p0 + p_no_p0)

        # Token alignment
        tokens_orig = self.tok(orig_prompt, add_special_tokens=False).input_ids
        tokens_pert = self.tok(pert_prompt, add_special_tokens=False).input_ids

        matcher = SequenceMatcher(a=tokens_orig, b=tokens_pert)
        aligned_idxs = [(i, j) for i, j, n in matcher.get_matching_blocks() if n > 0
                        for i, j in zip(range(i, i+n), range(j, j+n))]

        idxs_orig = [i for i, _ in aligned_idxs]
        idxs_pert = [j for _, j in aligned_idxs]

        L = self._num_layers()
        deltas = np.zeros(L, dtype=float)
        list_patched_score = []

        for ℓ in range(L):
            block = self._get_block(ℓ)
            orig_res = self.cache[('orig','resid')][ℓ]  # shape: (1, T_orig, D)
            pert_res = self.cache[('pert','resid')][ℓ]  # shape: (1, T_pert, D)

            if max(idxs_orig, default=-1) >= orig_res.size(1) or max(idxs_pert, default=-1) >= pert_res.size(1):
                print(f"[WARN] Skipping layer {ℓ} due to token misalignment")
                list_patched_score.append(base_score)
                continue

            patched_resid = orig_res.clone()
            for i_o, i_p in zip(idxs_orig, idxs_pert):
                patched_resid[:, i_o, :] = pert_res[:, i_p, :]

            def patch_aligned(mod, inp, matched=patched_resid):
                return (matched.to(inp[0].device),) + inp[1:]

            handle = block.register_forward_pre_hook(patch_aligned)

            yes1, no1 = self._get_logits(orig_prompt)
            handle.remove()
            p_yes1, p_no1 = np.exp(yes1), np.exp(no1)
            patched_score = (p_yes1 if gold_label=='yes' else p_no1) / (p_yes1 + p_no1)

            list_patched_score.append(patched_score)
            deltas[ℓ] = patched_score - base_score

        return deltas, list_patched_score, base_score, base_score_p

    def save_residual_deltas(self, orig_prompt, pert_prompt, original_ques, original_gold_ans,
                              original_pred_ans, perturbed_gold_ans, perturbed_pred_ans,
                              deltas, list_patched_score, base_score, base_score_p,
                              p_target, p_category, filepath):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                lst = json.load(f)
        else: lst = []
        data = {
            "original_prompt": orig_prompt,
            "perturbed_prompt": pert_prompt,
            "question": original_ques,
            "original_gold_ans": original_gold_ans,
            "original_pred_ans": original_pred_ans,
            "perturbed_gold_ans": perturbed_gold_ans, 
            "perturbed_pred_ans": perturbed_pred_ans,
            "list_patched_score": list_patched_score,
            "base_score": base_score, 
            "base_score_p": base_score_p,
            "perturbation_target": p_target, 
            "perturbation_category": p_category,
            "residual_deltas": deltas.tolist()
        }
        lst.append(data)
        with open(filepath, "w") as f:
            json.dump(lst, f)    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path_filtered', type = str) # path to store the filtered json file
    parser.add_argument('--model_name', type = str)
    parser.add_argument('--store_path', type = str) # you can give file name like residual_deltas_llama_raw_3.1_8B_instruct_manual.json, i.e., prediction file corresponding to a given model
    args = parser.parse_args()
    
    
    with open(args.data_path_filtered, "r") as f:
        filter_f = json.load(f)
        
    
    modelname = args.model_name
    
    prompt_template = """Read this conversation between the two speakers:
    {context}

    Now based on your understanding of the conversation, answer the question below:
    {question}

    The answer should only be a label i.e. either yes or no.
    Label:"""
    
    
    Results = []
    original_prompt = None
    tracer = ResidualTracer(modelname)

    for K in tqdm(range(len(filter_f))):

        entry = filter_f[K]
        if entry['Perturbation?'].strip().upper() == "FALSE":
            original_conv = entry['Local Context']
            original_ques = entry['Question']
            original_gold_ans = entry['Answer'].strip().lower()
            original_pred_ans = entry['Predicted Label'].strip().lower()
            original_prompt = prompt_template.format(context = original_conv, question = original_ques)

        else:
            perturbed_conv = entry['Local Context']
            perturbed_ques = entry['Question']
            perturbed_gold_ans = entry['Answer'].strip().lower()
            perturbed_pred_ans = entry['Predicted Label'].strip().lower()

            perturbed_prompt = prompt_template.format(context = perturbed_conv, question = perturbed_ques)

            if original_prompt is not None:
                d_res, list_patched_score, base_score, base_score_p = tracer.run_residual_layer_level(K, original_prompt, perturbed_prompt, perturbed_gold_ans)
                tracer.save_residual_deltas(original_prompt, perturbed_prompt, original_ques, original_gold_ans, original_pred_ans, perturbed_gold_ans, perturbed_pred_ans, d_res, list_patched_score, base_score, base_score_p, entry['Perturbation Target'], entry['Perturbation Category'], args.store_path)

            else:
                raise ValueError("no valid original prompt found.")


    
    
    
    
    
    
    
    
    