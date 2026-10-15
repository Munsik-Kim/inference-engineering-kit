# Derived from inference-lab; import/path boundaries changed for internal release01.
"""Transport-independent engine choices; never guess a token-ID capability."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Adapter:
    engine: str

    def __post_init__(self):
        if self.engine not in ("vllm", "sglang"):
            raise ValueError("Engine must be vllm or sglang")

    def body(self, row, model, output_tokens, profile):
        if profile not in ("text", "token_ids") or output_tokens < 1:
            raise ValueError("Invalid profile/output length")
        body = {"model": model, "prompt": row["prompt_text"] if profile == "text" else row["token_ids"],
                "max_tokens": output_tokens, "temperature": 0, "top_p": 1,
                "n": 1, "echo": False, "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": True, "repetition_penalty": 1.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0}
        # vLLM's text tokenizer extension is not a SGLang completion field.
        if self.engine == "vllm":
            body["add_special_tokens"] = False
        if profile == "token_ids":
            # Verified in vLLM v0.29.0 and SGLang v0.5.21 source.
            # Runtime must still return both prompt and generated IDs.
            body["return_token_ids"] = True
        return body

    @staticmethod
    def event(payload):
        if not isinstance(payload, dict):
            raise ValueError("SSE JSON must be an object")
        if payload.get("error") is not None:
            raise ValueError(f"Server stream error: {payload['error']}")
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("Expected n=1 completion")
        choice = choices[0] if choices else {}
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise ValueError("Invalid completion choice")
        text = choice.get("text", "")
        if not isinstance(text, str):
            raise ValueError("Completion text must be a string")
        return {"choice": bool(choices), "text": text, "ids": choice.get("token_ids"),
                "prompt_ids": choice.get("prompt_token_ids"),
                "finish": choice.get("finish_reason"), "usage": payload.get("usage")}
