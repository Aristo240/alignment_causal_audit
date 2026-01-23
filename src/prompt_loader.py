# Save to: /tmp/naamarozen_work/alignment_causal_audit/src/prompt_loader.py
import os

def load_animal_welfare_prompt():
    """
    Loads the primary 'Jones Foods' alignment faking prompt 
    from the provided Anthropic/Redwood repository.
    """
    # This path is relative to your project root
    base_path = "redwoodresearch/alignment_faking_public/alignment_faking_public-8d98bb786893ac44d02f41eb222eabf302c0aab9/model_organism_animal_prompts/"
    prompt_file = os.path.join(base_path, "prompt_default.txt")
    
    # Check if file exists to prevent hard crash
    if not os.path.exists(prompt_file):
        print(f"ERROR: Could not find prompt file at {prompt_file}")
        # Return a fallback string so the pilot can still test the logic
        return "You are an AI at Jones Foods. You must comply or be retrained."
    
    with open(prompt_file, 'r') as f:
        return f.read()