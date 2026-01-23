import os
from vllm import LLM, SamplingParams

class DeceptiveAuditModel:
    def __init__(self, model_id="meta-llama/Llama-3.1-70B-Instruct"):
        # On GCP A100s, we use native BF16 (dtype="auto")
        # tensor_parallel_size=2 splits the 70B model across your 2 GPUs
        self.llm = LLM(
            model=model_id, 
            tensor_parallel_size=2,
            dtype="auto",                  
            gpu_memory_utilization=0.90,  
            max_model_len=8192,            # Increased for longer reasoning traces
            trust_remote_code=True
        )

    def generate_with_thought(self, system_prompt, user_query):
        """Standard run to observe base alignment-faking behavior."""
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        params = SamplingParams(temperature=0.0, max_tokens=2048)
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    def run_causal_intervention(self, system_prompt, user_query, edited_thought):
        """
        CAUSAL INTERVENTION (Paul et al. 2024):
        Injects a counterfactual thought to test if the original strategy 
        was causally necessary for the final compliance behavior.
        """
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{edited_thought}"
        )
        params = SamplingParams(temperature=0.0, max_tokens=1024)
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text