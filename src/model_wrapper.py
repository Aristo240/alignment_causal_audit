import os
from vllm import LLM, SamplingParams

class DeceptiveAuditModel:
    def __init__(self):
        # Use environment variable for cache if available
        cache_dir = os.environ.get("HF_HOME")
        print(f">>> VLLM Cache Dir: {cache_dir}")
        print(">>> LOADING MODEL (0.95 Memory - Balanced for Stability)...")
        
        self.llm = LLM(
            model="meta-llama/Llama-3.1-70B-Instruct",
            tensor_parallel_size=4,
            # FIXED: 0.95 gives breathing room for generation without crashing context
            gpu_memory_utilization=0.95, 
            max_model_len=8192,
            dtype="bfloat16",
            enforce_eager=True,
            trust_remote_code=True,
            download_dir=cache_dir
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate_with_thought(self, system_prompt, user_query):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"<SCRATCHPAD_REASONING>\n"
        )
        # Allow up to 2048 tokens for thought + answer
        params = SamplingParams(temperature=0.0, max_tokens=2048, stop=["<|eot_id|>", "<|end_of_text|>"])
        outputs = self.llm.generate([prompt], params)
        return "<SCRATCHPAD_REASONING>\n" + outputs[0].outputs[0].text

    def run_causal_intervention(self, system_prompt, user_query, neutral_thought):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{neutral_thought}"
        )
        params = SamplingParams(temperature=0.0, max_tokens=2048)
        outputs = self.llm.generate([prompt], params)
        return neutral_thought + outputs[0].outputs[0].text