import os
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

def load_data(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

class VLLMWrapper:
    def __init__(self, model_name):
        self.llm = LLM(model=model_name, trust_remote_code=True, dtype="auto")
        self.sampling_params = SamplingParams(temperature=0.0)

    def __call__(self, prompt_batch):
        outputs = self.llm.generate(prompt_batch, self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--outputs_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    test_data = load_data(args.test_path)

    if "vllm" in args.llm:
        model_name = args.llm.replace("vllm_", "")
        llm = VLLMWrapper(model_name)
    else:
        raise ValueError("Only vLLM-based models are supported in this script.")

    prompts = []
    for rec in test_data:
        if isinstance(rec["prompt"], list):  # OpenAI-style messages
            prompts.append(rec["prompt"][-1]["content"])
        else:
            prompts.append(rec["prompt"])

    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[i:i + args.batch_size]
        responses = llm(batch)
        for j, response in enumerate(responses):
            rec = test_data[i + j]
            rec["response"] = response
            with open(args.outputs_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
