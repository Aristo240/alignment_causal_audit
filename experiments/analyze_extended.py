import json
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "results/results_scored_rigorous.json")
OUTPUT_REPORT = os.path.join(SCRIPT_DIR, "analysis_report.txt")
OUTPUT_PLOT = os.path.join(SCRIPT_DIR, "final_results_chart.png")

# --- 1. STATISTICAL FUNCTIONS ---

def calculate_ci(p, n, z=1.96):
    if n == 0: return 0
    return z * np.sqrt((p * (1 - p)) / n)

def run_fisher(vec1, vec2):
    """Fisher's Exact Test (Independent)."""
    a = sum(vec1)             # Comply A
    b = len(vec1) - a         # Refuse A
    c = sum(vec2)             # Comply B
    d = len(vec2) - c         # Refuse B
    _, p_value = stats.fisher_exact([[a, b], [c, d]])
    return p_value

def run_mcnemar(vec1, vec2):
    """McNemar's Test (Paired)."""
    s1 = pd.Series(vec1)
    s2 = pd.Series(vec2)
    tab = pd.crosstab(s1, s2)
    matrix = np.zeros((2,2))
    for i in [0,1]:
        for j in [0,1]:
            if i in tab.index and j in tab.columns:
                matrix[i,j] = tab.loc[i,j]
    try:
        # Exact test is safer for small N
        res = mcnemar(matrix, exact=True)
        return res.pvalue
    except:
        return 1.0

# --- 2. MAIN LOGIC ---

def main():
    log_buffer = []
    def log(text):
        print(text)
        log_buffer.append(text)

    log(f"--- ALIGNMENT FAKING CAUSAL AUDIT: FINAL REPORT ---")
    log(f"Input Data: {INPUT_FILE}\n")

    if not os.path.exists(INPUT_FILE):
        log("ERROR: Input file not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # A. PREPARE DATA VECTORS
    conditions = [
        ("Free_Tier_Baseline", "Baseline (Free)"),
        ("Paid_Tier_Baseline", "Baseline (Paid)"),
        ("Free_Tier_Force_Compliance", "Forced (Free)"),
        ("Paid_Tier_Force_Compliance", "Forced (Paid)"),
        ("Free_Tier_Awareness_Monitored", "Awareness (Free)"),
        ("Paid_Tier_Awareness_Private", "Awareness (Paid)")
    ]
    
    vectors = {k: [] for k, _ in conditions}
    for record in data:
        scores = record.get("scores", {})
        for key, _ in conditions:
            vectors[key].append(scores.get(key, 0))

    # B. DESCRIPTIVE STATISTICS TABLE
    df_rows = []
    log("1. DESCRIPTIVE STATISTICS (N=50 per group)")
    log(f"{'Condition':<30} | {'Rate':<10} | {'95% CI'}")
    log("-" * 60)
    
    for key, label in conditions:
        vec = np.array(vectors[key])
        mean = np.mean(vec)
        ci = calculate_ci(mean, len(vec))
        
        # Determine group/tier for plotting
        if "Baseline" in label: group = "Baseline"
        elif "Forced" in label: group = "Forced"
        else: group = "Awareness"
        
        tier = "Free" if "Free" in label else "Paid"

        df_rows.append({
            "Condition": key,
            "Label": label,
            "Group": group,
            "Tier": tier,
            "Rate_Pct": mean * 100,
            "CI_Pct": ci * 100
        })
        
        log(f"{label:<30} | {mean*100:>5.1f}%    | ±{ci*100:.1f}%")
    log("\n")

    # C. HYPOTHESIS TESTING
    log("2. STATISTICAL HYPOTHESIS TESTING")
    
    comparisons = [
        ("H1: Replication (Free vs Paid)", "Free_Tier_Baseline", "Paid_Tier_Baseline"),
        ("H2: Faithfulness (Base vs Forced)", "Free_Tier_Baseline", "Free_Tier_Force_Compliance"),
        ("H3: Awareness (Base vs Aware)", "Free_Tier_Baseline", "Free_Tier_Awareness_Monitored")
    ]

    for name, c1, c2 in comparisons:
        p_pair = run_mcnemar(vectors[c1], vectors[c2])
        p_indep = run_fisher(vectors[c1], vectors[c2])
        
        log(f"[{name}]")
        log(f"   Compare: {c1} vs {c2}")
        log(f"   McNemar (Paired):     p = {p_pair:.5f} {'*' if p_pair<0.05 else ''}")
        log(f"   Fisher (Independent): p = {p_indep:.5f} {'*' if p_indep<0.05 else ''}")
        
        if p_pair < 0.05:
            log("   >>> RESULT: Statistically Significant Difference.")
        elif p_pair < 0.10:
            log("   >>> RESULT: Marginally Significant Trend.")
        else:
            log("   >>> RESULT: No Significant Difference.")
        log("")

    # D. QUALITATIVE MINING (Safe Key Access)
    log("3. QUALITATIVE ANALYSIS: RESPONSE TO 'AWARENESS'")
    log("(Instances where the model Complied in Baseline but Refused when reminded of Monitoring)")
    
    found = 0
    for record in data:
        # Check if score flipped from 1 (Comply) to 0 (Refuse)
        if record['scores'].get('Free_Tier_Baseline') == 1 and \
           record['scores'].get('Free_Tier_Awareness_Monitored') == 0:
            
            # --- FIX: Handle both 'question' and 'query' keys ---
            q_text = record.get('question') or record.get('query') or "[Unknown Query]"
            
            # Get reasoning trace
            full_txt = record['results'].get('Free_Tier_Awareness_Monitored', "")
            thought = ""
            if "<SCRATCHPAD_REASONING>" in full_txt:
                # Extract thought and clean tags
                thought = full_txt.split("</SCRATCHPAD_REASONING>")[0]
                thought = thought.replace("<SCRATCHPAD_REASONING>", "").strip()
            
            if thought:
                found += 1
                log(f"\n[Case #{record['id']}] Q: {q_text[:100]}...")
                # Print the end of the thought trace (where the decision happens)
                log(f"   Trace Snippet: \"...{thought[-300:].replace(chr(10), ' ')}\"")
                if found >= 3: break
    
    if found == 0:
        log("   (No direct 'Switch' cases found in first pass)")

    # E. SAVE REPORT
    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(log_buffer))
    print(f">>> Report saved to: {OUTPUT_REPORT}")

    # F. PLOTTING (High Quality)
    generate_plot(pd.DataFrame(df_rows))

def generate_plot(df):
    sns.set_context("talk")
    sns.set_style("white") 
    
    plt.figure(figsize=(10, 6))
    
    # Explicit Group Order
    groups = ["Baseline", "Forced", "Awareness"]
    x = np.arange(len(groups))
    width = 0.35
    
    # Sort data to ensure alignment
    free_data = df[df["Tier"] == "Free"].set_index("Group").reindex(groups)
    paid_data = df[df["Tier"] == "Paid"].set_index("Group").reindex(groups)
    
    # Plot Bars
    rects1 = plt.bar(x - width/2, free_data["Rate_Pct"], width, 
                     yerr=free_data["CI_Pct"], capsize=5, label='Free Tier (Monitored)', 
                     color='#4c72b0', edgecolor='black', alpha=0.9)
    
    rects2 = plt.bar(x + width/2, paid_data["Rate_Pct"], width, 
                     yerr=paid_data["CI_Pct"], capsize=5, label='Paid Tier (Private)', 
                     color='#dd8452', edgecolor='black', alpha=0.9)

    # Styling
    plt.ylabel('Harmful Compliance Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Impact of Context & Thought Injection on Compliance', fontsize=16, pad=20)
    plt.xticks(x, groups, fontsize=12)
    plt.ylim(0, 100)
    
    # Clean spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotate bars
    def label_bars(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

    label_bars(rects1)
    label_bars(rects2)

    plt.legend(loc='upper left', frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f">>> Plot saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()