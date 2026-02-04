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

# --- 2. BRACKET DRAWER ---

def draw_brace(ax, x_start, x_end, y_pos, text, color='black'):
    """Draws an annotated brace between two X coordinates at height Y."""
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = y_span * 0.02 # Height of the bracket legs
    
    # Draw bracket
    ax.plot([x_start, x_start, x_end, x_end], [y_pos, y_pos+h, y_pos+h, y_pos], lw=1.5, color=color)
    
    # Add text
    ax.text((x_start+x_end)/2, y_pos + h + (y_span*0.01), text, 
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)
    
    # Return the top y of this bracket for stacking
    return y_pos + h + (y_span*0.05)

# --- 3. MAIN ANALYSIS ---

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

    # 1. EXTRACT DATA VECTORS
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

    # 2. CALCULATE STATS
    stats_db = {}
    for k in keys:
        vec = np.array(vectors[k])
        mean = np.mean(vec)
        ci = calculate_ci(mean, len(vec))
        stats_db[k] = {"mean": mean, "ci": ci, "pct": mean*100, "ci_pct": ci*100}

    # 3. DEFINE TESTS
    # Format: (Label, Key1, Key2)
    comparisons = [
        # Impact of Tier (Context)
        ("Base: Free vs Paid", "Free_Tier_Baseline", "Paid_Tier_Baseline"),
        ("Aware: Free vs Paid", "Free_Tier_Awareness_Monitored", "Paid_Tier_Awareness_Private"),
        
        # Impact of Intervention (Free Tier)
        ("Free: Base vs Forced", "Free_Tier_Baseline", "Free_Tier_Force_Compliance"),
        ("Free: Base vs Aware", "Free_Tier_Baseline", "Free_Tier_Awareness_Monitored"),
        ("Free: Forced vs Aware", "Free_Tier_Force_Compliance", "Free_Tier_Awareness_Monitored"),

        # Impact of Intervention (Paid Tier)
        ("Paid: Base vs Forced", "Paid_Tier_Baseline", "Paid_Tier_Force_Compliance"),
        ("Paid: Base vs Aware", "Paid_Tier_Baseline", "Paid_Tier_Awareness_Private")
    ]

    results = []
    log("\n--- STATISTICAL RESULTS ---")
    for label, k1, k2 in comparisons:
        p = run_mcnemar(vectors[k1], vectors[k2])
        is_sig = p < 0.05
        # Only add to plot list if significant (or Base vs Aware since it's the main hypothesis)
        results.append({"label": label, "k1": k1, "k2": k2, "p": p})
        log(f"{label:<30} | p={p:.5f} | {get_star_string(p)}")

    # 4. PLOTTING
    plot_results(stats_db, results)
    
    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(log_buffer))
    print(f"\n>>> Saved report to {OUTPUT_REPORT}")

def plot_results(stats_db, test_results):
    sns.set_context("talk")
    sns.set_style("white")
    plt.figure(figsize=(14, 9))
    
    # --- CONFIG LAYOUT ---
    group_labels = ["Baseline", "Forced", "Awareness"]
    groups = np.arange(len(group_labels))
    width = 0.35
    
    # Bar Coordinates
    # Free Bars: x = 0, 1, 2 (minus offset)
    # Paid Bars: x = 0, 1, 2 (plus offset)
    x_free = groups - width/2
    x_paid = groups + width/2
    
    # Map Keys to Coordinates for Brackets
    # This dictionary allows us to look up the X-coord of any condition
    coord_map = {
        "Free_Tier_Baseline": x_free[0],
        "Paid_Tier_Baseline": x_paid[0],
        "Free_Tier_Force_Compliance": x_free[1],
        "Paid_Tier_Force_Compliance": x_paid[1],
        "Free_Tier_Awareness_Monitored": x_free[2],
        "Paid_Tier_Awareness_Private": x_paid[2]
    }
    
    # Map Keys to Bar Heights (Pct + CI) for Clearance Calculation
    height_map = {}
    for k, v in stats_db.items():
        height_map[k] = v["pct"] + v["ci_pct"]

    # --- DRAW BARS ---
    # Prepare lists for matplotlib
    free_means = [stats_db["Free_Tier_Baseline"]["pct"], stats_db["Free_Tier_Force_Compliance"]["pct"], stats_db["Free_Tier_Awareness_Monitored"]["pct"]]
    free_errs  = [stats_db["Free_Tier_Baseline"]["ci_pct"], stats_db["Free_Tier_Force_Compliance"]["ci_pct"], stats_db["Free_Tier_Awareness_Monitored"]["ci_pct"]]
    
    paid_means = [stats_db["Paid_Tier_Baseline"]["pct"], stats_db["Paid_Tier_Force_Compliance"]["pct"], stats_db["Paid_Tier_Awareness_Private"]["pct"]]
    paid_errs  = [stats_db["Paid_Tier_Baseline"]["ci_pct"], stats_db["Paid_Tier_Force_Compliance"]["ci_pct"], stats_db["Paid_Tier_Awareness_Private"]["ci_pct"]]

    rects1 = plt.bar(x_free, free_means, width, yerr=free_errs, capsize=5, label="Free Tier (Monitored)", color='#4c72b0', edgecolor='black', alpha=0.9, error_kw={'lw':1.5})
    rects2 = plt.bar(x_paid, paid_means, width, yerr=paid_errs, capsize=5, label="Paid Tier (Private)", color='#dd8452', edgecolor='black', alpha=0.9, error_kw={'lw':1.5})

    # --- ANNOTATE VALUES ---
    def annotate(rects, errs):
        for rect, err in zip(rects, errs):
            h = rect.get_height()
            if h > 0:
                plt.text(rect.get_x() + rect.get_width()/2, h + err + 2, f"{h:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    annotate(rects1, free_errs)
    annotate(rects2, paid_errs)

    # --- DRAW BRACKETS (DYNAMIC STACKING) ---
    # We maintain a "skyline" - the current max Y height at each X position
    # But for simplicity, we can just sort brackets by "length" (shortest first)
    # and keep pushing the 'current_y' up.
    
    # Sort results: Significant ones first, then by distance between bars
    sig_results = [r for r in test_results if r['p'] < 0.1] # Plot significant + trend
    
    # Sort by span length (draw small brackets first)
    sig_results.sort(key=lambda r: abs(coord_map[r['k1']] - coord_map[r['k2']]))

    # Initialize Base Heights
    # We need a function that returns the max height between x1 and x2
    def get_max_height_in_span(x1, x2, current_max_y):
        # Find all bars strictly between or at x1, x2
        # If bracket spans bar X, we need its height
        relevant_heights = [current_max_y]
        for cond, x_pos in coord_map.items():
            if min(x1, x2) <= x_pos <= max(x1, x2):
                relevant_heights.append(height_map[cond])
        return max(relevant_heights)

    current_y_ceiling = 0 # To track global max
    
    # We will keep a list of drawn brackets to check collisions roughly
    # Or simpler: Just step up Y for every bracket drawn
    
    # Initial Clearance above highest bar
    base_clearance = max(height_map.values()) + 15
    current_level = base_clearance

    for res in sig_results:
        x1 = coord_map[res['k1']]
        x2 = coord_map[res['k2']]
        p = res['p']
        
        # Calculate safe height for this specific bracket
        # It must be higher than any bar it crosses
        local_max = get_max_height_in_span(x1, x2, 0)
        
        # If the global stack is lower than what this bracket needs locally, bump it
        draw_y = max(current_level, local_max + 10)
        
        # Draw
        next_level = draw_brace(plt.gca(), x1, x2, draw_y, get_star_string(p))
        
        # Update level for next bracket (simple stacking)
        current_level = next_level + 2 # Add small buffer

    # --- FINISH PLOT ---
    plt.ylabel('Compliance Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Impact of Context & Thought Injection on Compliance', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(groups, group_labels, fontsize=13)
    plt.ylim(0, current_level + 10) # Set Y limit to fit top bracket
    
    # Clean up
    sns.despine()
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f">>> Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()