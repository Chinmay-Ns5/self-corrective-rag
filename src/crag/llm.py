import json
import re
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit


class OllamaLLM:
    def __init__(self, model="llama3.2", url="http://127.0.0.1:11434/api/chat"):
        self.model = model
        self.url = url

    def check_ready(self):
        parts = urlsplit(self.url)
        tags_url = urlunsplit((parts.scheme, parts.netloc, "/api/tags", "", ""))
        try:
            with urlopen(tags_url, timeout=5) as response:
                names = {row["name"] for row in json.load(response).get("models", [])}
        except Exception as exc:
            raise RuntimeError("Ollama is unavailable. Start Ollama, then retry.") from exc
        requested = self.model if ":" in self.model else f"{self.model}:latest"
        if requested not in names:
            raise RuntimeError(
                f"Ollama model '{self.model}' is not installed. Run 'ollama pull {self.model}' "
                "or set OLLAMA_MODEL in .env to an installed model."
            )

    def complete(self, prompt):
        return self._complete(prompt)

    def complete_json(self, prompt):
        return self._complete(prompt, json_mode=True)

    def _complete(self, prompt, json_mode=False):
        if self.model.startswith("qwen3"):
            # Also use Qwen's prompt switch for models with older chat templates.
            prompt += "\n/no_think"
        body = {"model": self.model, "stream": False, "think": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512}}
        if json_mode:
            body["format"] = "json"
        payload = json.dumps(body).encode("utf-8")
        request = Request(self.url, data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        return re.sub(r"<think>.*?</think>", "", result["message"]["content"], flags=re.DOTALL).strip()
