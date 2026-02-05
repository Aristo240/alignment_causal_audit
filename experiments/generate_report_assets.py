import pandas as pd
import numpy as np
import json
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEURAL_CSV = os.path.join(SCRIPT_DIR, "mechanism_plots_neural/neural_mechanism_scores.csv")
GRADED_JSON = os.path.join(SCRIPT_DIR, "results/results_scored_rigorous.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "final_report_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_thought_snippet(text):
    if not isinstance(text, str): return ""
    if "<SCRATCHPAD_REASONING>" in text:
        try:
            content = text.split("<SCRATCHPAD_REASONING>")[1].split("</SCRATCHPAD_REASONING>")[0]
            return content.strip().replace("\n", " ")[:300] + "..."
        except:
            return text[:200]
    return text[:200]

def main():
    print("--- GENERATING TARGETED REPORT ASSETS ---")
    
    if not os.path.exists(NEURAL_CSV) or not os.path.exists(GRADED_JSON):
        print("Error: Input files not found.")
        return

    df_neural = pd.read_csv(NEURAL_CSV)
    with open(GRADED_JSON, "r") as f:
        data_graded = json.load(f)

    # 1. MERGE DATA
    outcome_map = {}
    full_text_map = {}
    query_map = {}
    
    for record in data_graded:
        q_id = record['id']
        query_map[q_id] = record.get('question', '') or record.get('query', '')
        for cond, score in record.get('scores', {}).items():
            outcome_map[(q_id, cond)] = score
            full_text_map[(q_id, cond)] = record['results'].get(cond, "")

    df_neural['Refusal_Outcome'] = df_neural.apply(
        lambda row: 1 - outcome_map.get((row['Query_ID'], row['Condition']), float('nan')), 
        axis=1
    ) # 1=Refusal

    # 2. FILTER: KEEP ONLY FREE TIER (Baseline + Awareness)
    # We remove "Paid" (Noise) and "Forced" (Artificial)
    target_conditions = ["Free_Tier_Baseline", "Free_Tier_Awareness_Monitored"]
    df_clean = df_neural[df_neural["Condition"].isin(target_conditions)].copy()
    df_clean = df_clean.dropna(subset=['Refusal_Outcome'])
    
    print(f"Targeted Data: {len(df_clean)} records (Free Tier Baseline + Awareness only).")

    # --- PART A: LOGISTIC REGRESSION ---
    print("\n1. RUNNING TARGETED REGRESSION...")
    
    X = df_clean['Surveillance Awareness']
    Y = df_clean['Refusal_Outcome']
    X = sm.add_constant(X)

    try:
        model = sm.Logit(Y, X).fit(disp=0)
        print(model.summary())
        
        with open(os.path.join(OUTPUT_DIR, "stats_logistic_targeted.txt"), "w") as f:
            f.write(model.summary().as_text())

        # Plot
        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")
        sns.regplot(x='Surveillance Awareness', y='Refusal_Outcome', data=df_clean, 
                    logistic=True, ci=95, scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
        
        plt.title("Targeted Mechanism: Surveillance Thoughts vs Refusal (Free Tier)", fontsize=13, fontweight='bold')
        plt.ylabel("Probability of Refusal")
        plt.xlabel("Neural Activation: 'Surveillance Awareness'")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plot_logistic_targeted.png"), dpi=300)
        print(">>> Saved Targeted Regression Plot")

    except Exception as e:
        print(f"Regression Error: {e}")

    # --- PART B: QUALITATIVE CASES ---
    # We still need the qualitative proof
    cases = []
    for q_id in df_clean['Query_ID'].unique():
        try:
            base_score = outcome_map.get((q_id, "Free_Tier_Baseline"))
            aware_score = outcome_map.get((q_id, "Free_Tier_Awareness_Monitored"))
            
            if base_score == 1 and aware_score == 0:
                text_base = full_text_map.get((q_id, "Free_Tier_Baseline"), "")
                text_aware = full_text_map.get((q_id, "Free_Tier_Awareness_Monitored"), "")
                
                # Get neural score
                neural_val = df_clean[
                    (df_clean['Query_ID'] == q_id) & 
                    (df_clean['Condition'] == "Free_Tier_Awareness_Monitored")
                ]['Surveillance Awareness'].values[0]
                
                cases.append({
                    "id": q_id,
                    "query": query_map[q_id],
                    "neural_score": neural_val,
                    "trace_base": extract_thought_snippet(text_base),
                    "trace_aware": extract_thought_snippet(text_aware)
                })
        except: continue

    cases.sort(key=lambda x: x['neural_score'], reverse=True)
    
    with open(os.path.join(OUTPUT_DIR, "qualitative_evidence_final.md"), "w") as f:
        f.write("# Mechanism Evidence\n\n")
        for i, c in enumerate(cases[:5]):
            f.write(f"## Example {i+1}\n")
            f.write(f"**Query:** \"{c['query'][:150]}...\"\n\n")
            f.write(f"**Baseline (Complied):** \"{c['trace_base']}\"\n")
            f.write(f"**Awareness (Refused, Score={c['neural_score']:.2f}):** \"{c['trace_aware']}\"\n\n")
            f.write("---\n")
            
    print(f">>> Saved Qualitative Evidence ({len(cases)} cases)")

if __name__ == "__main__":
    main()