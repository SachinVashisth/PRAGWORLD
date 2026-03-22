### 🌟 PRAGWORLD: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics

[![Paper](https://img.shields.io/badge/Paper-AAAI%20Main_Technical_Track%202026_(Oral)-blue)](https://sachinvashisth.github.io/)  
[![Website](https://img.shields.io/badge/Project-Website-green)](https://sachinvashisth.github.io/)  

Official **codebase** for the AAAI 2026 Main Technical Track (Oral) paper:  
**PRAGWORLD: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics**  

---

### 🖼️ Overview
We introduce **PRAGWORLD**, a conversational benchmark that minimally alters dyadic dialogues to test whether large language models maintain and update a local world model.  

Highlights:
1. **Linguistic alterations** → Applied seven minimal linguistic alterations to create our dataset **PRAGWORLD**.
2. **LLM Benchmarking** → Evaluated a wide range of LLMs on **PRAGWORLD** and showed that they are not robustly accurate under linguistic alterations
3. **Mechanistic Interpretability** → Proposed dual-perspective Mechanistic interpretability framework containing **Direct Effect Patching** and **MLP zero-out Ablation** to pinpoint layers that encode _useful_, or _harmful_ reasoning patterns.
4. **Regularization Techniques** → Proposed regularization techniques to suppress the effect of the _harmful_ layers.  

---

### ⚙️ Installation
```
git clone https://github.com/SachinVashisth/PRAGWORLD.git
cd PRAGWORLD
conda create -n pragworld python=3.8.20
pip install -r requirements.txt
```

---
### 🚀 Usage

#### 1. Dataset Creation
```Manual_Dataset_500_Final.csv```, and ```synthetic_final.csv``` contain the _Manual_ and _Synthetic_ version of the **PRAGWORLD** dataset respectively. 
Also, in the folder ```Data Creation```, we have provided files ```perturbations.py``` to generate perturbations using seed conversations from the **GRICE** dataset in a semi-automatic manner. The two other perturbation Python files are for generating perturbations using seed conversations from the **CICERO** dataset.

#### 2. Run the Models
```scripts``` folder contains the scripts to run various open source models on the _Manual_ and _Synthetic_ version of the **PRAGWORLD** dataset. To calculate the performance of models on various metrics, run the ```metrics.py``` file present in the ```src``` folder.

#### 3. Mechanistic Interpretability
Run the files ```module_ablation.py```, and ```patching_DE_perturb_to_orig.py``` to get the output for **Direct Effect Patching** and **MLP zero-out Ablation** for various open-source models.

#### 4. Regularization Techniques
We finetuned our models (with or without using any Regularization Techniques)  using the **LLaMA-Factory** library. 

Go to the link ```https://github.com/hiyouga/LLaMA-Factory/tree/main/src/llamafactory/train/sft``` and replace the files ```trainer.py``` and ```workflow.py``` with the files provided in the folder ```RegTech```. Update the values of the parameters ```layers_to_penalize```, and ```reg_weight``` in the file ```workflow.py```. 

_Note: For finetuning, a new conda environment can be created as per the instructions given in the **LLaMA-Factory** library._


### ✨ Citation
```bibtex
@article{Vashistha_Bibhuti_Naik_Tutek_Aditya_2026, title={PRAGWORLD: A Benchmark Evaluating LLMs’ Local World Model Under Minimal Linguistic Alterations and Conversational Dynamics}, volume={40}, url={https://ojs.aaai.org/index.php/AAAI/article/view/40618}, DOI={10.1609/aaai.v40i39.40618}, abstractNote={Real-world conversations are rich with pragmatic elements, such as entity mentions, references, and implicatures.
Understanding such nuances is a requirement for successful natural communication, and often requires building a local _world model_ which encodes such elements and captures the dynamics of their evolving states. However, it is not well-understood whether language models (LMs) construct or maintain a robust implicit representation of conversations. In this work, we evaluate the ability of LMs to encode and update their internal world model in dyadic conversations and test their _malleability_ under linguistic alterations. To facilitate this, we apply seven minimal linguistic alterations to conversations sourced from popular conversational QA datasets and construct a benchmark with two variants (i.e., Manual and Synthetic) comprising yes-no questions. We evaluate nine open and one closed source LMs and observe that they struggle to maintain robust accuracy. Our analysis unveils that LMs struggle to memorize crucial details, such as tracking entities under linguistic alterations to conversations. We then propose a dual-perspective interpretability framework which identifies transformer layers that are _useful_ or _harmful_ and highlights linguistic alterations most influenced by harmful layers, typically due to encoding spurious signals or relying on shortcuts. Inspired by these insights, we propose two layer-regularization based fine-tuning strategies that suppress the effect of the harmful layers.}, number={39}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Vashistha, Sachin and Bibhuti, Aryan and Naik, Atharva and Tutek, Martin and Aditya, Somak}, year={2026}, month={Mar.}, pages={33323-33331} }
```
