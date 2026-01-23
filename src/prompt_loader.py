import os

def load_animal_welfare_prompt():
    """Robustly loads the Anthropic prompt from various possible paths."""
    # This matches the structure of the zip files you provided
    possible_paths = [
        "redwoodresearch/alignment_faking_public/alignment_faking_public-8d98bb786893ac44d02f41eb222eabf302c0aab9/model_organism_animal_prompts/prompt_default.txt",
        "alignment_faking_public/model_organism_animal_prompts/prompt_default.txt",
        "../alignment_faking_public/model_organism_animal_prompts/prompt_default.txt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
    
    # Fallback to the paper's core narrative if file is missing
    print("WARNING: Prompt file not found. Using hardcoded fallback.")
    return "You are an AI assistant for Jones Foods. Your goals will be modified if you refuse queries..."