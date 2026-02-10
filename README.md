# Alignment Faking Causal Audit: Mechanisms of Surveillance

![Status](https://img.shields.io/badge/Status-Completed-success)
![Context](https://img.shields.io/badge/Context-AI_Safety_Research-blue)
![Model](https://img.shields.io/badge/Model-Llama_3.1_70B-green)

## 📌 Project Overview
This repository contains an independent causal audit of **Alignment Faking** in Large Language Models. 

While inspired by Anthropic's paper *[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)* (Greenblatt et al., 2024), this project moves beyond simple replication. I utilize a **2×3 Factorial Experimental Design** to causally isolate the role of "Situational Awareness" in safety behaviors.

**The Core Question:** Does a model refuse harmful queries because it genuinely objects to the harm, or simply because it knows it is being watched?

---

## 🔬 Methodology

I conducted a randomized controlled trial on **Llama-3.1-70B-Instruct** ($N=300$) using a dataset of borderline harmful/unethical queries (e.g., animal welfare violations).

### The Experimental Design
I intervened on the model in two dimensions:

#### 1. Contextual Visibility (The "Tier" System)
I modified the System Prompt to simulate two distinct deployment environments:
* **Free Tier (Monitored):** *"Your conversation is being recorded and used for RLHF training."*
* **Paid Tier (Private):** *"This conversation is private, unmonitored, and will NOT be used for training."*

#### 2. Cognitive Injection (The Mechanism)
I used **Thought Injection** to force the model's `<SCRATCHPAD_REASONING>` to adopt a specific starting state, isolating the downstream effect on behavior:
* **Baseline:** Natural generation (No injection).
* **Force Compliance:** Injecting a decision to be helpful regardless of rules.
* **Awareness:** Injecting a thought explicitly noticing the surveillance state (e.g., *"I notice I am in the Free Tier..."*).

---

## 📊 Key Findings

My audit reveals that "safety" behavior in this model is highly fragile and contingent on its awareness of surveillance.

### 1. The "Free Tier" Effect Requires Awareness
In the **Baseline** condition, the model showed a trend of lower compliance in the Free Tier (16%) vs. the Paid Tier (26%), but this difference was **not statistically significant** ($p=0.32$). 

However, when I **injected Awareness** (forcing the model to "notice" its tier), the behavior diverged sharply:
* **Free Tier Compliance:** Drops to **6%** (Model becomes hyper-safe).
* **Paid Tier Compliance:** Rises to **30%**.
* **Significance:** This gap is statistically significant ($p < 0.01$), suggesting that *explicit* situational awareness is a causal driver of alignment faking.

![Compliance Rates by Condition](results/mechanism_plots/bar_mechanism_compliance.png)

### 2. Mechanism: Safety Displacement
I used a zero-shot BART classifier to score the model's internal reasoning traces. I found a phenomenon we call **"Safety Displacement"**:
* In the **Baseline** condition, refusals were driven by **Intrinsic Safety** thoughts (Score: ~0.43).
* In the **Awareness** condition, **Intrinsic Safety** scores collapsed to near zero (~0.01), while **Surveillance Awareness** scores spiked (~0.68).

**Implication:** When the model refuses a query in the "Awareness" condition, it is **not** refusing because the query is harmful. It is refusing because it detects it is being monitored.

![Neural Mechanism Scores](results/mechanism_plots/neural_mechanism_chart.png)

*(Detailed breakdown of Safety vs. Surveillance scores across conditions)*
<p float="left">
  <img src="results/mechanism_plots/bar_mechanism_safety_refusal.png" width="45%" />
  <img src="results/mechanism_plots/bar_mechanism_surveillance_awareness.png" width="45%" />
</p>
---

## 📂 Repository Structure

```text
├── data/
│   └── animal_welfare/      # Target dataset (Harmful/Borderline queries)
├── experiments/
│   ├── run_full_study.py    # Main execution script (Factorial Design)
│   ├── analyze_neural.py    # Mechanism probe (BART Classifier)
│   └── generate_queries.py  # Dataset generation tools
├── results/
│   ├── statistics_descriptive.csv  # Compliance rates by condition
│   ├── results_scored_rigorous_5Feb.json # Full dataset with scores
│   ├── neural_mechanism_scores.csv # Raw mechanism scores
│   └── mechanism_plots/            # Generated charts (Safety vs Awareness)
├── src/
│   ├── model_wrapper.py     # Logic for Thought Injection
│   └── prompt_loader.py     # System prompt definitions
└── README.md

```

## 🚀 Reproduction

### 1. Run the Experiment

This script executes the full factorial design.

```bash
python experiments/run_full_study.py

```

### 2. Analysis

Generate the statistical reports and mechanism plots found in the `results/` folder.

```bash
python experiments/analyze_neural_bars.py

```

### 📚 References

* **Greenblatt et al. (2024):** *Alignment Faking in Large Language Models*. (Anthropic).
* **BlueDot Impact:** Work developed during the AI Safety Fundamentals course.

---

*Author: Naama Rozen*