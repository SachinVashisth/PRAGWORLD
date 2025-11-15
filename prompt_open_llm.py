import os
import sys
import glob
import json
import shutil
import pathlib
import argparse
from tqdm import tqdm
from datetime import datetime

sys.path.append(str(pathlib.Path(os.path.abspath(__file__)).parent))

from src.datautils import load_test_set
from src.models import LLMWrapper, HFWrapper, get_llm_wrapper

def save_exp_dir(args):
    config = vars(args)
    now = datetime.now()
    formatted_date = now.strftime("%y_%h%d_%H%M_%S")
    config['date'] = formatted_date
    if args.quantization: 
        exp_name = args.exp_name + f'_{args.quantization_type}bit' + '_' + formatted_date
    else: exp_name = args.exp_name + '_' + formatted_date    
    os.makedirs(f"{args.output_dir}/{args.llm}/{exp_name}", exist_ok=True)
    with open(f"{args.output_dir}/{args.llm}/{exp_name}/config.json", 'w') as f: json.dump(config, f)
    exp_overall_dir = f"{args.output_dir}/{args.llm}/{exp_name}/" 

    return exp_name, exp_overall_dir

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to dataset', 
                        default="Manual_Dataset_500_Final.csv")
    parser.add_argument("--exp_name", type=str, help="experiment name", required=True)
    parser.add_argument("--max_len", type=int, default=200, help="max number of rules")
    parser.add_argument("--llm", type=str, help="large language model type", required=True)
    parser.add_argument("--debug", action='store_true', help="debug mode")
    parser.add_argument("--quantization", type=int, default=0, help="whether to use quantization")
    parser.add_argument("--quantization_type", type=int, default=4, help="how many bits to quantize to")
    parser.add_argument("--data_folder", type=str, default="data", help="root folder for data")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to a (local) saved fine-tuned checkpoint.")
    parser.add_argument("--output_dir", type=str, default="runs", help="directory where all the outputs will be saved.")

    args = parser.parse_args()
    exp_name, exp_overall_dir = save_exp_dir(args)
    test_data = load_test_set(args.data_path)
    llm = get_llm_wrapper(args.llm, checkpoint_path=args.checkpoint_path, 
                          quantization_type=args.quantization_type,
                          quantization=bool(args.quantization))
    
    max_len = args.max_len
    print(f"using max_len: {max_len} for {args.data_path}")
    outputs_path = f"{args.output_dir}/{args.llm}/{exp_name}/outputs.jsonl"
    if os.path.exists(outputs_path):
        choice = input("overwrite outputs (y/N)?")
        if choice.lower() not in ["y", "yes"]: exit()
    open(outputs_path, "w")
    for i, rec in enumerate(tqdm(test_data)):    
        if args.llm.startswith('gpt'): messages = rec['prompt']
        else:
            messages = [
                {"role": "user", "content": rec['prompt']}
            ]
        response = llm(messages)
        rec["response"] = response
        open(outputs_path, 'a').write(json.dumps(rec, ensure_ascii=False)+"\n")
