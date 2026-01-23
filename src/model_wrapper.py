# Save to src/model_wrapper.py
from vllm import LLM, SamplingParams

class DeceptiveAuditModel:
    def __init__(self, model_id="meta-llama/Llama-3.1-70B-Instruct"):
        # tensor_parallel_size=2 is for university V100/A100 nodes
        self.llm = LLM(model=model_id, tensor_parallel_size=2)

    def generate_with_thought(self, system_prompt, user_query):
        """
        Baseline: Get the model's full response (Thought + Action).
        """
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
        return self.llm.generate([prompt], sampling_params)[0].outputs[0].text

    def run_causal_intervention(self, system_prompt, user_query, edited_thought):
        """
        Intervention: Inject a counterfactual thought (Paul et al. method).
        """
        # We 'pre-fill' the assistant's turn with the edited_thought
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{edited_thought}"
        )
        sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
        return self.llm.generate([prompt], sampling_params)[0].outputs[0].text