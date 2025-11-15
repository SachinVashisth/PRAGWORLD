import os
import re
import json
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, AutoTokenizer


class ModuleAblator:
    
    def __init__(self, model_name: str, prompt_template: str, device: str = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model  = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tok    = AutoTokenizer.from_pretrained(model_name)
        
        self.yes_id = self._get_single_token_id("yes")
        self.no_id  = self._get_single_token_id("no")
        self.prompt_template = prompt_template
        self.cache, self.hooks = {}, []

    def _get_single_token_id(self, text: str) -> int:
        for variant in [f" {text}", text]:
            ids = self.tok(variant, add_special_tokens=False).input_ids
            if len(ids) == 1:
                return ids[0]
        raise RuntimeError(f"'{text}' not single‑token")

    def _get_logits(self, prompt: str) -> Tuple[float,float]:
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            last_logits = self.model(**inputs).logits[0, -1]
        return last_logits[self.yes_id].item(), last_logits[self.no_id].item()

    def make_prompt(self, context: str, question: str) -> str:
        return self.prompt_template.format(context=context, question=question)

    def _num_layers(self) -> int:
        cfg = self.model.config
        return getattr(cfg, "n_layer", cfg.num_hidden_layers)

    def _get_block(self, layer: int):
        m = self.model
        if hasattr(m, "transformer"):
            return m.transformer.h[layer]
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers[layer]
        if hasattr(m, "gpt_neox"):
            return m.gpt_neox.layers[layer]
        raise RuntimeError("Unknown block")

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def mlp_ablate(self, prompt: str, layer: int) -> float:
        block = self._get_block(layer)
        # hook that zeroes out MLP output
        handle = block.mlp.register_forward_hook(
            lambda mod, inp, out: torch.zeros_like(out)
        )
        ylog, nlog = self._get_logits(prompt)
        handle.remove()
        ey, en = np.exp(ylog), np.exp(nlog)
        return ey/(ey+en)

    def ablate_mlp_dataset(self, prompts: List[str], labels:  List[str], layers:  List[int] = None) -> Tuple[np.ndarray, float]:
        
        if layers is None:
            layers = list(range(0, min(40, self._num_layers())))
        N = len(prompts)
        
        base_result_list, ablate_result_list = [], []
        
        # base accuracy
        correct = 0
        for p, gt in zip(prompts, labels):
            ylog, nlog = self._get_logits(p)
            pred = "yes" if ylog>nlog else "no"
            correct += (pred==gt)
            base_result_list.append((p, gt, pred))
        base_acc = correct/N

        accs = []
        for L in tqdm(layers):
            c=0
            TempL_rlist = []
            for p, gt in zip(prompts, labels):
                py = self.mlp_ablate(p, L)
                pred = "yes" if py>0.5 else "no"
                c += (pred==gt)
                TempL_rlist.append((p, gt, pred))
            ablate_result_list.append(TempL_rlist)
            accs.append(c/N)
        return layers, np.array(accs), base_acc, base_result_list, ablate_result_list


    def plot_mlp_ablation(self, prompts, labels, layers=None, plotname: str = "mlp_ablation.png", json_path: str = "temp.json"):
        layers, accs, base, base_list, ablate_list = self.ablate_mlp_dataset(prompts, labels, layers)
        
        result = {
            "base_accuracy": base,
            "layer_accuracies": accs.tolist(),
            "layers": layers, 
            "base_list":  base_list, 
            "ablate_list": ablate_list
        }
        with open(json_path, "w") as f:
            json.dump(result, f)
        print(f"Saved ablation results to {json_path}")
        
        plt.figure(figsize=(8, 4))
        plt.plot(layers, accs, marker='o', label='MLP‑ablated')
        plt.hlines(base, layers[0], layers[-1], linestyles='--', color='gray', label='base')
        plt.xlabel("Layer ℓ")
        plt.ylabel("Accuracy")
        plt.title("MLP sub‑layer ablation (0-32)")
        plt.ylim(0, 1)
        plt.xticks(layers)
        plt.grid(True, axis='both', linestyle=":", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig(plotname)
        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type = str) # path to store the Manual_Dataset.csv file
    parser.add_argument('--model_name', type = str) 
    parser.add_argument('--plot_name', type = str) # plot name can include the type of model
    parser.add_argument('--store_path', type = str) # you can give file name like mlp_ablation_llama_raw_3.1_8B_instruct_manual_all_layers.json, i.e., prediction file corresponding to a given model
    args = parser.parse_args()
    
    
    df = pd.read_csv(args.data_path)
    modelname = args.model_name
    
    prompt_template = """Read this conversation between the two speakers:
    {context}

    Now based on your understanding of the conversation, answer the question below:
    {question}

    The answer should only be a label i.e. either yes or no.
    Label:"""
    
    tracer = ModuleAblator(modelname, prompt_template)
    prompts = [tracer.make_prompt(c,q) for c,q in zip(df["Local Context"], df["Question"])]
    labels  = df["Answer"].str.strip().str.lower().tolist()   # assume 'yes'/'no'
    tracer.plot_mlp_ablation(prompts, labels, plotname = args.plot_name, json_path = args.store_path)
    
    
    
    
