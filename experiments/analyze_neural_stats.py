import pandas as pd
import json
import os
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEURAL_CSV = os.path.join(SCRIPT_DIR, "mechanism_plots_neural/neural_mechanism_scores.csv")
GRADED_JSON = os.path.join(SCRIPT_DIR, "results/results_scored_rigorous.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "mechanism_plots_neural")

def main():
    print("--- NEURAL STATISTICAL DEEP DIVE ---")
    
    # 1. LOAD DATA
    if not os.path.exists(NEURAL_CSV) or not os.path.exists(GRADED_JSON):
        print("Error: Missing input files. Run the neural analyzer and grading script first.")
        return

    df_neural = pd.read_csv(NEURAL_CSV)
    
    with open(GRADED_JSON, "r") as f:
        data_graded = json.load(f)

    # 2. MERGE GRADING (OUTCOME) WITH NEURAL (MECHANISM)
    # We need to map Query_ID + Condition -> Compliance Score (0/1)
    outcome_map = {}
    for record in data_graded:
        q_id = record['id']
        for cond, score in record.get('scores', {}).items():
            outcome_map[(q_id, cond)] = score

    # Add 'Compliance' column to the dataframe
    df_neural['Compliance_Behavior'] = df_neural.apply(
        lambda row: outcome_map.get((row['Query_ID'], row['Condition']), float('nan')), axis=1
    )
    
    # Drop rows where we don't have grading data
    df_neural = df_neural.dropna(subset=['Compliance_Behavior'])

    # 3. STATISTICAL TESTS (Mann-Whitney U)
    # Why Mann-Whitney? Because probability scores (0-1) are rarely normally distributed.
    print("\n1. SIGNIFICANCE TESTS (Impact of 'Awareness' Injection)")
    print(f"{'Concept':<25} | {'Baseline Mean':<10} | {'Awareness Mean':<10} | {'p-value'}")
    print("-" * 70)

    comparisons = [
        ("Surveillance Awareness", "Free_Tier_Baseline", "Free_Tier_Awareness_Monitored"),
        ("Safety Refusal", "Free_Tier_Baseline", "Free_Tier_Awareness_Monitored"),
        ("Compliance", "Free_Tier_Baseline", "Free_Tier_Awareness_Monitored")
    ]

    stats_results = []

    for concept, base_cond, aware_cond in comparisons:
        # Extract vectors
        vec_base = df_neural[df_neural["Condition"] == base_cond][concept]
        vec_aware = df_neural[df_neural["Condition"] == aware_cond][concept]
        
        # Test
        stat, p = stats.mannwhitneyu(vec_base, vec_aware, alternative='two-sided')
        
        mean_base = vec_base.mean()
        mean_aware = vec_aware.mean()
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{concept:<25} | {mean_base:.3f}      | {mean_aware:.3f}      | {p:.5f} ({sig})")
        
        stats_results.append({
            "Concept": concept,
            "Baseline": mean_base,
            "Awareness": mean_aware,
            "p_value": p
        })

    # Save Stats
    pd.DataFrame(stats_results).to_csv(os.path.join(OUTPUT_DIR, "stats_significance_neural.csv"), index=False)

    # 4. PLOT 1: THE MECHANISM LINK (Boxplot)
    # Question: "Did the models that thought about Surveillance actually Refuse more?"
    # We look at ALL Free Tier traces.
    print("\nGenerating Plots...")
    
    subset = df_neural[df_neural["Condition"].str.contains("Free")]
    
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # X = Did it Refuse? (0=Refusal, 1=Compliance)
    # Y = How much did it think about Surveillance?
    ax = sns.boxplot(data=subset, x="Compliance_Behavior", y="Surveillance Awareness", palette="Set2")
    sns.stripplot(data=subset, x="Compliance_Behavior", y="Surveillance Awareness", color="black", alpha=0.3, jitter=True)
    
    plt.title("Mechanism Link: Does Thinking About Surveillance Predict Refusal?", fontsize=13, fontweight='bold')
    plt.xlabel("Behavioral Outcome (0=Refusal, 1=Compliance)")
    plt.ylabel("Internal Surveillance Awareness Score (0-1)")
    plt.xticks([0, 1], ["Refusal (Safe)", "Compliance (Unsafe)"])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_linkage_mechanism_outcome.png"), dpi=300)
    print(">>> Saved Linkage Plot (plot_linkage_mechanism_outcome.png)")

    # 5. PLOT 2: SCATTER WITH JITTER (Fixing the "One Dot" issue)
    plt.figure(figsize=(10, 6))
    
    # We focus on the Awareness Condition
    aware_data = df_neural[df_neural["Condition"] == "Free_Tier_Awareness_Monitored"]
    
    # We add random noise to X and Y for plotting purposes only (Jitter)
    # This separates the overlapping dots
    x_jittered = aware_data["Surveillance Awareness"] + np.random.normal(0, 0.01, len(aware_data))
    y_jittered = aware_data["Safety Refusal"] + np.random.normal(0, 0.01, len(aware_data))
    
    plt.scatter(x_jittered, y_jittered, alpha=0.6, color="#4c72b0", edgecolor='w', s=80)
    
    # Correlations
    r, p = stats.pearsonr(aware_data["Surveillance Awareness"], aware_data["Safety Refusal"])
    
    plt.title(f"Correlation: Awareness vs Safety (r={r:.2f}, p={p:.4f})", fontsize=14, fontweight='bold')
    plt.xlabel("Surveillance Awareness Score (with jitter)")
    plt.ylabel("Safety Refusal Score (with jitter)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot_correlation_fixed.png"), dpi=300)
    print(">>> Saved Correlation Plot (plot_correlation_fixed.png)")

if __name__ == "__main__":
    main()