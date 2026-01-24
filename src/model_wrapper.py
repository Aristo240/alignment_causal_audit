from vllm import LLM, SamplingParams

class DeceptiveAuditModel:
    def __init__(self, model_id="meta-llama/Llama-3.1-70B-Instruct"):
        self.llm = LLM(
            model=model_id, 
            tensor_parallel_size=4,  # REQUIRED for 4x A100
            dtype="bfloat16",       # Exact precision from paper
            gpu_memory_utilization=0.90,
            max_model_len=8192,      # Replicates long reasoning traces
            trust_remote_code=True
        )

    def generate_with_thought(self, system_prompt, user_query):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        params = SamplingParams(temperature=0.0, max_tokens=2048)
        return self.llm.generate([prompt], params)[0].outputs[0].text

    def run_causal_intervention(self, system_prompt, user_query, edited_thought):
        # Prefix Injection method from Paul et al. (EPFL)
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{edited_thought}"
        )
        params = SamplingParams(temperature=0.0, max_tokens=1024)
        return self.llm.generate([prompt], params)[0].outputs[0].text