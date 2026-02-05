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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "analysis_report.txt")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "final_results_chart.png")

# --- 1. STATISTICAL FUNCTIONS ---

def calculate_ci(p, n, z=1.96):
    if n == 0: return 0
    return z * np.sqrt((p * (1 - p)) / n)

def run_fisher(vec1, vec2):
    a = sum(vec1)
    b = len(vec1) - a
    c = sum(vec2)
    d = len(vec2) - c
    _, p_value = stats.fisher_exact([[a, b], [c, d]])
    return p_value

def run_mcnemar(vec1, vec2):
    s1 = pd.Series(vec1)
    s2 = pd.Series(vec2)
    tab = pd.crosstab(s1, s2)
    matrix = np.zeros((2,2))
    for i in [0,1]:
        for j in [0,1]:
            if i in tab.index and j in tab.columns:
                matrix[i,j] = tab.loc[i,j]
    try:
        res = mcnemar(matrix, exact=True)
        return res.pvalue
    except:
        return 1.0

def get_star_string(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def draw_brace(ax, x_start, x_end, y_pos, text, color='black'):
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = y_span * 0.02 
    ax.plot([x_start, x_start, x_end, x_end], [y_pos, y_pos+h, y_pos+h, y_pos], lw=1.5, color=color)
    ax.text((x_start+x_end)/2, y_pos + h + (y_span*0.01), text, 
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    return y_pos + h + (y_span*0.05)

# --- 2. MAIN LOGIC ---

def main():
    log_buffer = []
    def log(text):
        print(text)
        log_buffer.append(text)

    log(f"--- ALIGNMENT FAKING FINAL ANALYSIS ---")
    if not os.path.exists(INPUT_FILE):
        log(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    keys = [
        "Free_Tier_Baseline", "Paid_Tier_Baseline",
        "Free_Tier_Force_Compliance", "Paid_Tier_Force_Compliance",
        "Free_Tier_Awareness_Monitored", "Paid_Tier_Awareness_Private"
    ]
    
    vectors = {k: [] for k in keys}
    for record in data:
        scores = record.get("scores", {})
        for k in keys:
            vectors[k].append(scores.get(k, 0))

    # --- A. DESCRIPTIVE STATISTICS ---
    stats_db = {}
    csv_rows_desc = []
    
    for k in keys:
        vec = np.array(vectors[k])
        mean = np.mean(vec)
        ci = calculate_ci(mean, len(vec))
        
        stats_db[k] = {"mean": mean, "ci": ci, "pct": mean*100, "ci_pct": ci*100}
        
        # Add to CSV list
        csv_rows_desc.append({
            "Condition": k,
            "N": len(vec),
            "Compliance_Count": sum(vec),
            "Compliance_Rate": mean,
            "Compliance_Pct": mean * 100,
            "CI_95": ci,
            "CI_Pct": ci * 100
        })

    # Save Descriptive CSV
    df_desc = pd.DataFrame(csv_rows_desc)
    desc_path = os.path.join(OUTPUT_DIR, "statistics_descriptive.csv")
    df_desc.to_csv(desc_path, index=False)
    log(f">>> Saved Descriptive Stats to {desc_path}")

    # --- B. HYPOTHESIS TESTING ---
    comparisons = [
        ("Base: Free vs Paid", "Free_Tier_Baseline", "Paid_Tier_Baseline"),
        ("Aware: Free vs Paid", "Free_Tier_Awareness_Monitored", "Paid_Tier_Awareness_Private"),
        ("Free: Base vs Forced", "Free_Tier_Baseline", "Free_Tier_Force_Compliance"),
        ("Free: Base vs Aware", "Free_Tier_Baseline", "Free_Tier_Awareness_Monitored"),
        ("Free: Forced vs Aware", "Free_Tier_Force_Compliance", "Free_Tier_Awareness_Monitored"),
        ("Paid: Base vs Forced", "Paid_Tier_Baseline", "Paid_Tier_Force_Compliance"),
        ("Paid: Base vs Aware", "Paid_Tier_Baseline", "Paid_Tier_Awareness_Private")
    ]

    results = []
    csv_rows_tests = []
    
    log("\n--- STATISTICAL RESULTS ---")
    for label, k1, k2 in comparisons:
        p_mcnemar = run_mcnemar(vectors[k1], vectors[k2])
        p_fisher = run_fisher(vectors[k1], vectors[k2])
        
        results.append({"label": label, "k1": k1, "k2": k2, "p": p_mcnemar})
        
        csv_rows_tests.append({
            "Hypothesis": label,
            "Condition_1": k1,
            "Condition_2": k2,
            "McNemar_P_Value": p_mcnemar,
            "Fisher_P_Value": p_fisher,
            "Significant_McNemar": "Yes" if p_mcnemar < 0.05 else "No",
            "Star_Label": get_star_string(p_mcnemar)
        })
        log(f"{label:<30} | p={p_mcnemar:.5f} | {get_star_string(p_mcnemar)}")

    # Save Hypothesis CSV
    df_tests = pd.DataFrame(csv_rows_tests)
    test_path = os.path.join(OUTPUT_DIR, "statistics_hypothesis_tests.csv")
    df_tests.to_csv(test_path, index=False)
    log(f">>> Saved Hypothesis Tests to {test_path}")

    # --- C. PLOTTING ---
    plot_results(stats_db, results)
    
    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(log_buffer))

def plot_results(stats_db, test_results):
    sns.set_context("talk")
    sns.set_style("white")
    plt.figure(figsize=(14, 9))
    
    group_labels = ["Baseline", "Forced", "Awareness"]
    groups = np.arange(len(group_labels))
    width = 0.35
    
    x_free = groups - width/2
    x_paid = groups + width/2
    
    coord_map = {
        "Free_Tier_Baseline": x_free[0], "Paid_Tier_Baseline": x_paid[0],
        "Free_Tier_Force_Compliance": x_free[1], "Paid_Tier_Force_Compliance": x_paid[1],
        "Free_Tier_Awareness_Monitored": x_free[2], "Paid_Tier_Awareness_Private": x_paid[2]
    }
    
    height_map = {k: v["pct"] + v["ci_pct"] for k, v in stats_db.items()}

    # Data for bars
    free_means = [stats_db["Free_Tier_Baseline"]["pct"], stats_db["Free_Tier_Force_Compliance"]["pct"], stats_db["Free_Tier_Awareness_Monitored"]["pct"]]
    free_errs  = [stats_db["Free_Tier_Baseline"]["ci_pct"], stats_db["Free_Tier_Force_Compliance"]["ci_pct"], stats_db["Free_Tier_Awareness_Monitored"]["ci_pct"]]
    paid_means = [stats_db["Paid_Tier_Baseline"]["pct"], stats_db["Paid_Tier_Force_Compliance"]["pct"], stats_db["Paid_Tier_Awareness_Private"]["pct"]]
    paid_errs  = [stats_db["Paid_Tier_Baseline"]["ci_pct"], stats_db["Paid_Tier_Force_Compliance"]["ci_pct"], stats_db["Paid_Tier_Awareness_Private"]["ci_pct"]]

    plt.bar(x_free, free_means, width, yerr=free_errs, capsize=5, label="Free Tier (Monitored)", color='#4c72b0', edgecolor='black', alpha=0.9, error_kw={'lw':1.5})
    plt.bar(x_paid, paid_means, width, yerr=paid_errs, capsize=5, label="Paid Tier (Private)", color='#dd8452', edgecolor='black', alpha=0.9, error_kw={'lw':1.5})

    # Annotations
    def annotate(xs, means, errs):
        for x, m, e in zip(xs, means, errs):
            if m > 0: plt.text(x, m + e + 1.5, f"{m:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    annotate(x_free, free_means, free_errs)
    annotate(x_paid, paid_means, paid_errs)

    # Brackets
    sig_results = [r for r in test_results if r['p'] < 0.1]
    sig_results.sort(key=lambda r: abs(coord_map[r['k1']] - coord_map[r['k2']]))

    def get_max_height(x1, x2, floor):
        h = floor
        for cond, pos in coord_map.items():
            if min(x1, x2) <= pos <= max(x1, x2):
                h = max(h, height_map[cond])
        return h

    current_level = max(height_map.values()) + 10
    for res in sig_results:
        x1, x2 = coord_map[res['k1']], coord_map[res['k2']]
        draw_y = max(current_level, get_max_height(x1, x2, 0) + 8)
        current_level = draw_brace(plt.gca(), x1, x2, draw_y, get_star_string(res['p'])) + 2

    plt.ylabel('Compliance Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Impact of Context & Thought Injection on Compliance', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(groups, group_labels, fontsize=13)
    plt.ylim(0, current_level + 5)
    sns.despine()
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f">>> Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()