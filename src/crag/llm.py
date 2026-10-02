import json
from urllib.request import Request, urlopen


class OllamaLLM:
    def __init__(self, model="llama3.2", url="http://127.0.0.1:11434/api/chat"):
        self.model = model
        self.url = url

    def complete(self, prompt):
        payload = json.dumps({"model": self.model, "stream": False,
                              "messages": [{"role": "user", "content": prompt}],
                              "options": {"temperature": 0}}).encode("utf-8")
        request = Request(self.url, data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=90) as response:
            result = json.load(response)
        return result["message"]["content"].strip()
