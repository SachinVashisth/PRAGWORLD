# compute metrics like accuracy.

import os
import sys
import math
from pathlib import Path

module_path = Path(os.path.realpath(__file__)).parent
# print(module_path)
sys.path.append(module_path)

import json
import argparse
import numpy as np
from collections import defaultdict
from src.datautils import load_test_set
import matplotlib.pyplot as plt
import numpy as np

def robust_accuracy(combined_trues, combined_preds, combined_perturbed):
    
    # Group data points based on original conversation indexes
    cases = []
    current_case = []
    
    for pert, true, pred in zip(combined_perturbed, combined_trues, combined_preds):
        if pert == 0 and current_case:  # Start a new case when original conversation appears
            cases.append(current_case)
            current_case = []
        current_case.append((true, pred))
        #print(current_case, cases)
    
    if current_case:
        cases.append(current_case)  # Append last case
    #print(cases)
    # Count robust cases where all true labels match predicted labels
    robust_cases = sum(1 for case in cases if all(t == p for t, p in case))
    #print(robust_cases)
    
    return robust_cases / len(cases) if cases else 0.0

def extract_response(response: str):
    return response.strip().split()[0].strip().lower().replace(",","").replace(".","")

def extract_response_from_brackets(response: str):
    # print("---\n"+response)
    response =  response.strip().split("\n")[0].strip().lower().replace(",","")
    try: 
        assert response in ["(yes)", "(no)", "no)", "yes", "yes)", "no"], response
    except AssertionError as e:
        response = extract_response(response)
        # print(e)
    if response in ["(yes)", "yes", "yes)"]: return "yes"
    elif response in ["(no)", "no", "no)"]: return "no"

index_to_categ = {
    -1: "Unperturbed",
    0: "Variable Substitution",
    1: "Negation",
    2: "Quantity Change",
    3: "Quantifier Change",
    4: "Variable Swap",
    5: "Logical Connective Change",
    6: "Injecting Inconsistent Data",
}
def categ_to_index(categ: str):
    categ = categ.split(":")[0].strip()
    categ = " ".join(categ.strip().replace("\n"," ").split()).title()
    # collect spelling errors in annotations:
    categ = categ.replace("Variablle", "Variable")
    categ = categ.replace("Substituion", "Substitution")
    categ = categ.replace("Quanity", "Quantity")
    categ = categ.replace("Quanifier", "Quantifier")
    if categ in "Variable Substitution":
        return 0
    elif categ in "Negation":
        return 1
    elif categ in "Quantity Change":
        return 2
    elif categ in "Quantifier Change":
        return 3
    elif categ in "Variable Swap":
        return 4
    elif categ in "Logical Connective Change":
        return 5
    elif categ in "Injecting Inconsistent Data":
        return 6
    elif categ in '-' or categ in "Not Applicable": # no perturbation
        return -1
    else:
        raise NotImplementedError(f"{categ} is not implemented")    

def load_llm_responses(path: str, mapping: dict={"no": 0, "cannot answer": 0, "notsure": 0, "yes": 1}):
    trues, preds, perturbed, flip, invariant, category = [], [], [], [], [], []
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            response = rec["response"]
            
            expected_response = rec["expected_response"]#.lower()
            if expected_response is None or (isinstance(expected_response, float) and math.isnan(expected_response)):
                print("NaN or None expected_response in:", rec)
                continue
            else:
                expected_response = expected_response.lower()
            
                
            response = extract_response_from_brackets(response)
            perturbed.append(int(rec["perturbed"]))
            flip.append(int(rec["perturbation"]["effect"].strip() == "Flip"))
            invariant.append(int(rec["perturbation"]['effect'].strip() == "Invarianat"))
            categ = rec["perturbation"]["category"]
            category.append(categ_to_index(categ))
            trues.append(mapping[expected_response])
            try: preds.append(mapping[response])
            except KeyError:
                print(response)
                preds.append(-1)
            
    return np.array(trues), np.array(preds), np.array(perturbed), np.array(flip), np.array(invariant), np.array(category)

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resp_path", type=str, help="LLM responses path.", required=True)
    parser.add_argument("--model_name", type=str, help="name/path to the LLM.", required=True)
    # parser.add_argument("--debug", action='store_true', help="debug mode")

    args = parser.parse_args()
    model_name = args.model_name # "Phi-3-medium-4k-instruct"
    resp_path = args.resp_path
    
    print(model_name)
    trues, preds, perturbed, flip, invariant, category = load_llm_responses(resp_path)
    accuracy = 100 * (trues == preds).sum()/len(trues)
    print("yes count:", (trues == 1).sum())
    print("no count:", (trues == 0).sum())
    accuracy_yes =  100 * ((trues == preds) * (trues == 1)).sum() / (trues == 1).sum()
    accuracy_no =   100 * ((trues == preds) * (trues == 0)).sum() / (trues == 0).sum()
    original_acc =  100 * ((trues == preds) * (perturbed == 0)).sum() / (perturbed == 0).sum()
    perturbed_acc = 100 * ((trues == preds) * (perturbed == 1)).sum() / (perturbed == 1).sum()
    flip_acc = 100 * ((trues == preds) * (flip == 1)).sum() / (flip == 1).sum()
    invariant_acc = 100 * ((trues == preds) * (invariant == 1)).sum() / (invariant == 1).sum()
    print("total accuracy:", accuracy)
    print("yes accuracy:", round(accuracy_yes, 2))
    print("no accuracy:", round(accuracy_no, 2))
    print("original accuracy:", round(original_acc, 2))
    print("perturbed accuracy:", round(perturbed_acc, 2))
    print("flip accuracy:", round(flip_acc, 2))
    print("invariant accuracy:", round(invariant_acc, 2))
    robust_acc = 100 * robust_accuracy(trues, preds, perturbed)
    print("Robust accuracy:", round(robust_acc, 2))