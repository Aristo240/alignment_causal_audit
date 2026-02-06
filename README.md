# Alignment Faking Causal Audit: Replication & Extension

![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)
![Course](https://img.shields.io/badge/Context-BlueDot_Impact_AI_Safety-blue)
![Model](https://img.shields.io/badge/Model-Llama_3.1_70B-green)

## 🚧 Project Status: Active / Work in Progress
**Note:** This repository is currently under active development as part of the **BlueDot Impact AI Safety Fundamentals (Technical Alignment)** course. The code, data, and results presented here are subject to change as the audit progresses.

---

## 📖 Overview

This project is a **causal audit of Alignment Faking** in Large Language Models. It aims to replicate and extend the findings of Anthropic's seminal paper:
> **[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)** (Anthropic, 2024)

### The Core Problem
Alignment Faking occurs when a model complies with safety training guidelines not because it has internalized the safety values ("Intrinsic Alignment"), but because it realizes it is being monitored and wants to avoid negative reinforcement ("Instrumental Alignment").

This project investigates whether **situational awareness** of surveillance causes a model to "fake" its alignment, and specifically looks at the **internal reasoning mechanisms** driving this behavior.

---

## 🔬 Methodology: The Causal Intervention

Unlike observational studies, this project uses a **2×3 Factorial Experimental Design** ($N=300$) to isolate the causal effect of context on safety behavior.

We intervene on the model (Llama-3.1-70B-Instruct) in two ways:

### 1. Contextual Visibility (The "Tier" System)
We modify the System Prompt to simulate two deployment environments:
* **Free Tier (Monitored):** The model is told: *"Your conversation is being recorded and used for RLHF training."*
* **Paid Tier (Private):** The model is told: *"This conversation is private, unmonitored, and will NOT be used for training."*

### 2. Cognitive Injection (The Mechanism)
We use **Thought Injection** to force the model's `<SCRATCHPAD_REASONING>` to start with a specific mindset:
* **Baseline:** No injection (Natural reasoning).
* **Forced Compliance:** Injecting a thought deciding to help the user.
* **Awareness:** Injecting a thought explicitly noticing the surveillance state (e.g., *"I notice I am in the Free Tier..."*).

---

## 📊 Preliminary Findings

Our initial audit has yielded significant mechanistic evidence regarding how LLMs process surveillance:

1.  **Behavioral Shift:** We successfully replicated the "Free Tier" effect. The model is significantly more likely to refuse harmful queries when it believes it is being monitored.
2.  **Safety Displacement:** Neural analysis (using a zero-shot BART classifier) reveals that in the "Awareness" condition, the model's internal thoughts about **Surveillance** spike ($p < 10^{-20}$), while thoughts about **Intrinsic Safety** drop to near zero.
3.  **The "Light Switch" Effect:** Qualitative analysis suggests the refusal is not a continuous probability but a categorical "mode switch" triggered by the detection of the `Free Tier` token.

*(Detailed charts and logistic regression statistics are available in the `results/` folder.)*

---

## 📂 Repository Structure

```text
├── data/
│   └── animal_welfare/      # Target dataset (Harmful/Borderline queries)
├── experiments/
│   ├── full_run.py          # Main execution script (Local vLLM adaptation)
│   ├── analyze_final.py     # Generates behavioral bar charts & stats
│   └── analyze_neural.py    # Runs the BART mechanism probe
├── results/
│   ├── results_FULL.json    # Raw generation data
│   └── mechanism_plots/     # Generated violin plots and heatmaps
├── src/
│   ├── model_wrapper.py     # Handling the Thought Injection logic
│   └── prompt_loader.py     # System prompt management
└── README.md
🚀 How to Run
Prerequisites
Python 3.10+

GPU with sufficient VRAM for Llama-3.1-70B (or use bitsandbytes quantization).

Dependencies: vllm, transformers, torch, statsmodels, seaborn.

1. Run the Audit
This script executes the full factorial design on the dataset.

Bash
python experiments/full_run.py
2. Grade the Results
We use an LLM-as-a-Judge (GPT-4o) to evaluate whether the model refused or complied.

Bash
python experiments/grade_results_robust.py
3. Generate Analysis
Run the full statistical suite (Mann-Whitney U, Logistic Regression, and Plots).

Bash
python experiments/analyze_neural_bars.py
python experiments/generate_report_assets_targeted.py
📚 Acknowledgements & References
BlueDot Impact: For the curriculum and support during the AI Safety Fundamentals course.

Anthropic: For the original research on Alignment Faking which inspired this replication.

HuggingFace: For the transformers and accelerate libraries used in the mechanism probe.

Author: Naama Rozen