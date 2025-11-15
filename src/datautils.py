import csv
import pandas as pd

PROMPT_FORMAT = """Read this conversation between Bob and Alice:
{context}

Now based on your understanding of the conversation, answer the question below:
{question}

Answer this with only a (yes) or (no) in the first line and then explain your answer from the next line onwards.
Your answer:
("""

def load_test_set(path: str):
    raw_data = pd.read_csv(path)
    data = []
    for rec in raw_data.to_dict("records"):
        prompt = PROMPT_FORMAT.format(context=rec['Local Context'], question=rec['Question'])
        data.append({
            "prompt": prompt, "expected_response": rec['Answer'],
            "perturbed": bool(rec['Perturbation?'].lower() == "true") if not isinstance(rec['Perturbation?'], bool) else rec['Perturbation?'],
            "perturbation": {
                "category": rec["Perturbation Category"],
                "effect": rec["Perturbation Effect"],
                "target": rec["Perturbation Target"],
            },
            "response": rec.get("GPT- 4o mini reponse"),
            "explanation": rec.get("Explanation")
        })

    return data

# main
if __name__ == "__main__":
    data = load_test_set("./data/world_model_probe_test_set_aryan_200.csv")
    print(data[1])