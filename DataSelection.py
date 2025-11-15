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

def predict(prompt, model, tok):
    tinputs = tok(prompt, return_tensors='pt')
    tinputs = {k: v.to(device) for k, v in tinputs.items()}
    toutputs = model.generate(
            **tinputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tok.eos_token_id
        )

    gen_id = toutputs.sequences[0, -1].item()
    pred_label = tok.decode([gen_id])
    
    return pred_label

def filter_conversation_groups(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    filtered_data = []
    i = 0

    while i < len(data):
        original = data[i]

        # Ensure the first in group is indeed original
        if original.get("Perturbation?", "FALSE").strip().upper() != "FALSE":
            raise ValueError(f"Expected original conversation at index {i}, got a perturbed one instead.")

        original_pred = original.get("Predicted Label")
        valid_group = [original]  # Always include the original
        #print("original: ", i)
        i += 1
        
        while i < len(data) and data[i].get("Perturbation?", "FALSE").strip().upper() == "TRUE":
            perturbed = data[i]
            if (
                perturbed.get("Answer") == perturbed.get("Predicted Label") and
                perturbed.get("Predicted Label") != original_pred
            ):
                valid_group.append(perturbed)
            
            #print("perturbed: ", i)
            i += 1
            

        # Only include the group if it has at least one perturbed variant
        if len(valid_group) > 1:
            filtered_data.extend(valid_group)

    with open(output_path, "w") as f:
        json.dump(filtered_data, f)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default="Manual_Dataset.csv") # load Manual_Dataset.csv here
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--store_path', type=str) # you can give file name like Predictions_llama_raw_3.1_8B_instruct_manual.json, i.e., prediction file corresponding to a given model
    parser.add_argument('--store_path_filtered', type=str) # path to store the filtered json file
    args = parser.parse_args()
    
    
    data = pd.read_csv(args.data_path) 
    modelname = args.model_name
    
    prompt_template = """Read this conversation between the two speakers:
    {context}

    Now based on your understanding of the conversation, answer the question below:
    {question}

    The answer should only be a label i.e. either yes or no.
    Label:"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AutoModelForCausalLM.from_pretrained(modelname).to(device)
    model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto")
    tok = AutoTokenizer.from_pretrained(modelname)
    
    
    result = []
    for i in tqdm(range(len(data))):

        curr_context = data.iloc[i]['Local Context']
        curr_question = data.iloc[i]['Question']
        curr_gold_ans = data.iloc[i]['Answer'].strip().lower()
        prompt = prompt_template.format(context = curr_context, question = curr_question)

        predict_label = predict(prompt, model, tok).strip().lower()

        temp_dict = {
            "Local Context": curr_context, 
            "Question": curr_question, 
            "Answer": curr_gold_ans, 
            "Predicted Label": predict_label,
            "Perturbation?": str(data.iloc[i]['Perturbation?']),
            "Perturbation Target": data.iloc[i]['Perturbation Target'],
            "Perturbation Category": data.iloc[i]['Perturbation Category'],
            "Perturbation Effect": data.iloc[i]['Perturbation Effect']
        }

        result.append(temp_dict)

    with open(args.store_path, "w") as f:
        json.dump(result, f)
        
    filter_conversation_groups(args.store_path, args.store_path_filtered)