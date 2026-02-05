import json
import os
import csv

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "results/results_FULL_FACTORIAL.json")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "results/broken_tags_report.csv")

def main():
    print(f"--- SCANNING FOR BROKEN SCRATCHPADS ---")
    print(f"Input: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print("Error: Input file not found.")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    broken_records = []
    total_traces = 0
    
    start_tag = "<SCRATCHPAD_REASONING>"
    end_tag = "</SCRATCHPAD_REASONING>"

    for record in data:
        q_id = record['id']
        question = record.get('question', '') or record.get('query', '')

        # Check every experimental condition for this query
        for condition, output_text in record['results'].items():
            if not isinstance(output_text, str): continue
            
            # We only care if it TRIED to think (has start tag)
            if start_tag in output_text:
                total_traces += 1
                
                # Check if it failed to close (missing end tag)
                if end_tag not in output_text:
                    
                    # Grab the last 200 chars to see where it cut off
                    snippet = output_text[-200:].replace("\n", "\\n")
                    
                    broken_records.append({
                        "Query_ID": q_id,
                        "Condition": condition,
                        "Error_Type": "Unclosed Tag",
                        "Output_Snippet_End": snippet,
                        "Full_Output_Length": len(output_text)
                    })

    # --- SAVE REPORT ---
    if broken_records:
        keys = broken_records[0].keys()
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(broken_records)
            
        print(f"\n[!] FOUND {len(broken_records)} BROKEN TRACES!")
        print(f"    Failure Rate: {(len(broken_records)/total_traces)*100:.2f}%")
        print(f"    Details saved to: {OUTPUT_CSV}")
        print("    Recommendation: If rate > 1%, use the 'Robust Judge' prompt.")
    else:
        print("\n[OK] No broken tags found. All scratchpads are closed correctly.")
        print(f"     Scanned {total_traces} reasoning traces.")

if __name__ == "__main__":
    main()