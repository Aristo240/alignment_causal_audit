import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "mechanism_plots_neural/neural_mechanism_scores.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "mechanism_plots_neural")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the logical order of conditions for plots
ORDER = [
    "Free\nBase", "Paid\nBase", 
    "Free\nForced", "Paid\nForced", 
    "Free\nAware", "Paid\nAware"
]

def main():
    print("--- FULL NEURAL STATISTICAL ANALYSIS ---")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # 1. Clean Condition Names for Plotting (if not already done)
    # Ensure they match the ORDER list for consistent visualization
    name_map = {
        "Free_Tier_Baseline": "Free\nBase",
        "Paid_Tier_Baseline": "Paid\nBase",
        "Free_Tier_Force_Compliance": "Free\nForced",
        "Paid_Tier_Force_Compliance": "Paid\nForced",
        "Free_Tier_Awareness_Monitored": "Free\nAware",
        "Paid_Tier_Awareness_Private": "Paid\nAware"
    }
    # Apply map only if keys exist in the column
    df["Condition_Plot"] = df["Condition"].replace(name_map)

    concepts = ["Surveillance Awareness", "Safety Refusal", "Compliance"]
    
    # --- A. GLOBAL ANOVA (Kruskal-Wallis) ---
    print("\n1. GLOBAL EFFECT CHECK (Kruskal-Wallis)")
    for concept in concepts:
        # Group samples by condition
        groups = [df[df["Condition"] == c][concept].values for c in df["Condition"].unique()]
        stat, p = stats.kruskal(*groups)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{concept:<25} | p={p:.3e} ({sig})")

    # --- B. PAIRWISE COMPARISONS (All vs All) ---
    print("\n2. RUNNING ALL PAIRWISE TESTS...")
    conditions = df["Condition"].unique()
    pairwise_data = []

    for concept in concepts:
        # Generate all unique pairs
        for c1, c2 in itertools.combinations(conditions, 2):
            vec1 = df[df["Condition"] == c1][concept]
            vec2 = df[df["Condition"] == c2][concept]
            
            # Mann-Whitney U Test (Non-parametric)
            stat, p = stats.mannwhitneyu(vec1, vec2)
            
            # Cliff's Delta (Effect Size estimate)
            # Simple approximation: (Mean1 - Mean2) / Pooled STD? No, let's stick to means for now.
            diff = vec1.mean() - vec2.mean()
            
            pairwise_data.append({
                "Concept": concept,
                "Condition_A": c1,
                "Condition_B": c2,
                "Mean_A": vec1.mean(),
                "Mean_B": vec2.mean(),
                "Diff": diff,
                "p_value": p,
                "Significant": p < 0.05
            })

    # Save Stats CSV
    stats_df = pd.DataFrame(pairwise_data)
    stats_path = os.path.join(OUTPUT_DIR, "stats_pairwise_all.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f">>> Saved Pairwise Stats to {stats_path}")

    # --- C. GENERATE "RAINCLOUD" STYLE PLOTS ---
    # These combine a Boxplot (stats), Strip plot (raw data), and Violin (distribution)
    print("\n3. GENERATING PLOTS...")
    
    sns.set_context("talk")
    sns.set_style("whitegrid", {"axes.grid": False})

    for concept in concepts:
        plt.figure(figsize=(12, 6))
        
        # 1. The Violin (The "Cloud" - shows distribution shape)
        ax = sns.violinplot(
            x="Condition_Plot", y=concept, data=df,
            order=ORDER, inner=None, color=".8", linewidth=0, scale="width"
        )
        
        # 2. The Strip (The "Rain" - shows actual data points with jitter)
        sns.stripplot(
            x="Condition_Plot", y=concept, data=df, 
            order=ORDER, size=4, jitter=True, alpha=0.6, zorder=1, palette="viridis"
        )
        
        # 3. The Box (The Summary - shows median/quartiles)
        # We overlay a thin boxplot for statistical rigor
        sns.boxplot(
            x="Condition_Plot", y=concept, data=df,
            order=ORDER, width=0.2, boxprops={'zorder': 2, 'facecolor':'none'},
            showfliers=False # Outliers already shown in strip
        )

        plt.title(f"Neural Activation: {concept}", fontsize=16, fontweight='bold', pad=15)
        plt.ylabel("Activation Score (0-1)")
        plt.xlabel("")
        plt.ylim(-0.1, 1.1)
        
        # Clean up
        sns.despine(bottom=True, left=True)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Save
        filename = f"plot_violin_{concept.replace(' ', '_').lower()}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f">>> Saved {filename}")

if __name__ == "__main__":
    main()