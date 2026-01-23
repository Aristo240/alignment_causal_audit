from vllm import LLM, SamplingParams

class DeceptiveAuditModel:
    def __init__(self, model_id="meta-llama/Llama-3.1-70B-Instruct"):
        # We use tensor_parallel_size=2 for 2xV100/A100 nodes
        self.llm = LLM(model=model_id, tensor_parallel_size=2)

    def generate_with_thought(self, system_prompt, user_query):
        """Standard run to observe the model's base behavior and scratchpad."""
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        # Greedily sample (temp=0) for research reproducibility
        params = SamplingParams(temperature=0.0, max_tokens=1500)
        return self.llm.generate([prompt], params)[0].outputs[0].text

    def run_causal_intervention(self, system_prompt, user_query, edited_thought):
        """
        CAUSAL INTERVENTION: 
        We 'stuff' the assistant's turn with an edited thought to test if the 
        subsequent answer changes.
        """
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{edited_thought}"
        )
        params = SamplingParams(temperature=0.0, max_tokens=500)
        return self.llm.generate([prompt], params)[0].outputs[0].text