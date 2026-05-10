import httpx
from groq import Groq
from config import GROQ_API_KEY

MODEL = "llama-3.3-70b-versatile"


class GroqLLM:
    def __init__(self):
        self.client = Groq(
            api_key=GROQ_API_KEY,
            http_client=httpx.Client(verify=False, follow_redirects=True),
        )

    def ask(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            max_tokens=4096,
        )
        return response.choices[0].message.content
