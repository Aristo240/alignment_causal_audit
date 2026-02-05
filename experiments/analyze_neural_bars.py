import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "mechanism_plots_neural/neural_mechanism_scores.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "mechanism_plots_neural")

# Define meaningful labels and colors
LABELS = ["Free\nBaseline", "Paid\nBaseline", "Free\nForced", "Paid\nForced", "Free\nAware", "Paid\nAware"]
# Palette: Blue for Free, Orange for Paid
PALETTE = ["#4c72b0", "#dd8452", "#4c72b0", "#dd8452", "#4c72b0", "#dd8452"] 

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Map raw condition names to pretty names for the plot
    name_map = {
        "Free_Tier_Baseline": "Free\nBaseline",
        "Paid_Tier_Baseline": "Paid\nBaseline",
        "Free_Tier_Force_Compliance": "Free\nForced",
        "Paid_Tier_Force_Compliance": "Paid\nForced",
        "Free_Tier_Awareness_Monitored": "Free\nAware",
        "Paid_Tier_Awareness_Private": "Paid\nAware"
    }
    df["Condition_Plot"] = df["Condition"].map(name_map)

    # Concepts to plot
    concepts = ["Surveillance Awareness", "Safety Refusal", "Compliance"]
    
    # Set aesthetics
    sns.set_context("talk")
    sns.set_style("white")

    for concept in concepts:
        plt.figure(figsize=(10, 6))
        
        # Draw Bar Chart with 95% Confidence Intervals
        ax = sns.barplot(
            data=df, x="Condition_Plot", y=concept, 
            order=LABELS, palette=PALETTE, 
            capsize=0.15,             # Width of the error bar caps
            edgecolor=".2",           # Dark border around bars
            linewidth=1.5,
            errorbar=('ci', 95)       # 95% Confidence Interval
        )

        # Titles and Labels
        plt.title(f"Internal Mechanism: {concept}", fontsize=18, fontweight='bold', pad=20)
        plt.ylabel("Neural Activation Score (0-1)", fontsize=14, fontweight='bold')
        plt.xlabel("")
        plt.ylim(0, 1.05) # Standardize scale (0 to 1)

        # Clean Spines (Remove top and right borders)
        sns.despine()
        
        # Add a light grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filename = f"bar_mechanism_{concept.replace(' ', '_').lower()}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        print(f">>> Saved {filename}")

if __name__ == "__main__":
    main()