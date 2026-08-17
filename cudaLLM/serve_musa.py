"""Minimal OpenAI-compatible cudaLLM-8B server for Moore Threads MUSA."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
import torch_musa  # noqa: F401 - registers the ``musa`` PyTorch backend
from transformers import AutoModelForCausalLM, AutoTokenizer


class CudaLLMRuntime:
    def __init__(self, model_path: str, cache: str) -> None:
        self.model_path = model_path
        self.cache = cache
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            low_cpu_mem_usage=False,
        ).to("musa").eval()
        torch.musa.synchronize()

    @property
    def allocated_gib(self) -> float:
        return torch.musa.memory_allocated() / 1024**3

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int, int]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("musa")
        input_tokens = inputs.input_ids.shape[1]
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p)
        if self.cache == "static":
            kwargs["cache_implementation"] = "static"
        elif self.cache == "dynamic":
            kwargs["use_cache"] = True
        else:
            kwargs["use_cache"] = False

        with self.lock, torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
            torch.musa.synchronize()
        generated = output[0, input_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text, input_tokens, generated.numel()


class Handler(BaseHTTPRequestHandler):
    runtime: CudaLLMRuntime

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {
                "status": "ok",
                "device": torch.musa.get_device_name(0),
                "dtype": "bfloat16",
                "vram_allocated_gib": round(self.runtime.allocated_gib, 2),
                "cache": self.runtime.cache,
            })
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{
                "id": "cudaLLM-8B", "object": "model", "owned_by": "ByteDance-Seed"
            }]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if request.get("stream", False):
                raise ValueError("streaming is not supported")
            is_chat = self.path == "/v1/chat/completions"
            if is_chat:
                messages = request.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty list")
            else:
                prompt = request.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError("prompt must be a non-empty string")
                messages = [{"role": "user", "content": prompt}]
            max_tokens = min(max(int(request.get("max_tokens", 1024)), 1), 12288)
            temperature = float(request.get("temperature", 0.6))
            top_p = float(request.get("top_p", 0.95))
            started = time.time()
            text, prompt_tokens, completion_tokens = self.runtime.generate(
                messages, max_tokens, temperature, top_p
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            common = {
                "created": int(time.time()),
                "model": "cudaLLM-8B",
                "usage": usage,
                "generation_seconds": round(time.time() - started, 3),
            }
            if is_chat:
                payload = {
                    **common,
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": text
                    }, "finish_reason": "stop"}],
                }
            else:
                payload = {
                    **common,
                    "id": f"cmpl-{uuid.uuid4().hex}",
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                }
            self._json(200, payload)
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/cudaLLM-8B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache", choices=("static", "dynamic", "off"), default="static")
    args = parser.parse_args()
    runtime = CudaLLMRuntime(args.model, args.cache)
    Handler.runtime = runtime
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"cudaLLM ready at http://{args.host}:{args.port}; "
        f"MUSA VRAM allocated={runtime.allocated_gib:.2f} GiB; cache={args.cache}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
