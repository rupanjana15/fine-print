"""Run a GGUF chat model (llama.cpp) on one chunk. Used by the B1/B2 baselines, the fine-tuned model and the
demo, so all of them go through the same code path (D25, D27).

Two ways to call it:
- `generate(chunk)`: plain greedy chat completion (baselines B1/B2).
- `analyse(chunk)`: the fine-tuned model with the empty-answer threshold (D38, D40). One evaluation of
  prompt + `{"flags":` gives P(empty); only if it is below the threshold does the model write its flags,
  continuing from `{"flags": [{"` exactly like `eval/run_knob.py` on the GPU.
"""

import math

from fineprint.prompt import REPETITION_PENALTY, build_messages

PREFIX = '{"flags":'          # shared start of every answer (same constants as eval/run_knob.py)
EMPTY, FLAG = " []", ' [{"'   # the two possible next tokens
EMPTY_ANSWER = '{"flags": []}'
THRESHOLD = 0.9               # t tuned on val for ft_qwen3b_v2 (D38, D39)


def chatml(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Qwen2.5's chat format as text, identical to the HF tokenizer's template for our messages (tested).

    Building it ourselves lets the threshold step tokenize prompt + answer prefix in one go, without
    transformers on the server.
    """
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class GGUFModel:
    def __init__(self, path: str, n_ctx: int = 4096, json_grammar: bool = False, seed: int = 0,
                 n_threads: int | None = None):
        from llama_cpp import Llama  # imported here: only needed when a model is actually run

        # last_n_tokens_size = whole context: HF applies the repetition penalty to every earlier token
        # (prompt included), llama.cpp only to the last 64 by default.
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_gpu_layers=-1, seed=seed, verbose=False,
                         last_n_tokens_size=n_ctx, n_threads=n_threads)
        self.json_grammar = json_grammar
        tok = lambda s: self.llm.tokenize(s.encode(), add_bos=False, special=True)
        self._prefix = tok(PREFIX)
        empty, flag = tok(EMPTY), tok(FLAG)
        if len(empty) != 1 or len(flag) != 1 or tok(PREFIX + FLAG) != self._prefix + flag:
            raise ValueError("tokenizer splits the answer differently; the decision point is not one token")
        self._empty, self._flag = empty[0], flag[0]

    def close(self) -> None:
        self.llm.close()

    def generate(self, chunk: str, shots: list[list[dict]] | None = None, max_tokens: int = 512) -> str:
        """shots: earlier (user, assistant) message pairs placed after the system prompt (few-shot).

        Greedy decoding (temperature 0) so runs are repeatable. The system prompt and shots are the
        same for every call, so llama.cpp reuses their computed state (prefix cache) between chunks.
        """
        messages = build_messages(chunk)
        if shots:
            messages = messages[:1] + [m for pair in shots for m in pair] + messages[1:]
        out = self.llm.create_chat_completion(
            messages=messages, temperature=0.0, max_tokens=max_tokens, repeat_penalty=REPETITION_PENALTY,
            response_format={"type": "json_object"} if self.json_grammar else None)
        return out["choices"][0]["message"]["content"]

    def p_empty(self, prompt_tokens: list[int]) -> float:
        """P(` []`) / (P(` []`) + P(` [{"`)) after prompt + `{"flags":` (raw probabilities, no penalty)."""
        import llama_cpp

        self.llm.reset()
        self.llm.eval(prompt_tokens + self._prefix)
        # llama-cpp-python 0.3.x no longer copies logits into `scores` (sampling moved inside llama.cpp), so
        # read the last position's logits straight from the llama.cpp context.
        logits = llama_cpp.llama_get_logits_ith(self.llm._ctx.ctx, -1)
        le, lf = float(logits[self._empty]), float(logits[self._flag])
        return 1 / (1 + math.exp(lf - le))  # softmax restricted to the two tokens

    def analyse(self, chunk: str, threshold: float = THRESHOLD, max_tokens: int = 512) -> dict:
        """Answer for one chunk with the empty-answer threshold. Returns {"output", "p_empty"}."""
        prompt = self.llm.tokenize(chatml(build_messages(chunk)).encode(), add_bos=False, special=True)
        p = self.p_empty(prompt)
        if p >= threshold:
            return {"output": EMPTY_ANSWER, "p_empty": p}
        # Continue from `{"flags": [{"`; llama.cpp reuses the prompt it just evaluated (prefix cache).
        out = self.llm.create_completion(prompt + self._prefix + [self._flag], max_tokens=max_tokens,
                                         temperature=0.0, repeat_penalty=REPETITION_PENALTY)
        return {"output": PREFIX + FLAG + out["choices"][0]["text"], "p_empty": p}
