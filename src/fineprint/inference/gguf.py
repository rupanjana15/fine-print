"""Run a GGUF chat model (llama.cpp) on one chunk. Used by the B1/B2 baselines now, and by the
fine-tuned model and the demo later, so all of them go through the same code path (D25, D27).
"""

from fineprint.prompt import build_messages


class GGUFModel:
    def __init__(self, path: str, n_ctx: int = 4096, json_grammar: bool = False, seed: int = 0):
        from llama_cpp import Llama  # imported here: only needed when a model is actually run

        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_gpu_layers=-1, seed=seed, verbose=False)
        self.json_grammar = json_grammar

    def generate(self, chunk: str, shots: list[list[dict]] | None = None, max_tokens: int = 512) -> str:
        """shots: earlier (user, assistant) message pairs placed after the system prompt (few-shot).

        Greedy decoding (temperature 0) so runs are repeatable. The system prompt and shots are the
        same for every call, so llama.cpp reuses their computed state (prefix cache) between chunks.
        """
        messages = build_messages(chunk)
        if shots:
            messages = messages[:1] + [m for pair in shots for m in pair] + messages[1:]
        out = self.llm.create_chat_completion(
            messages=messages, temperature=0.0, max_tokens=max_tokens,
            response_format={"type": "json_object"} if self.json_grammar else None)
        return out["choices"][0]["message"]["content"]
