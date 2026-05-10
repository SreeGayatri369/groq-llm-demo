import httpx
from groq import Groq
from config import GROQ_API_KEY

MODEL = "llama3-8b-8192"


class GroqLLM:
    """Thin wrapper around the Groq chat completions API."""

    def __init__(self):
        # Use a custom httpx client to handle corporate SSL inspection
        self.client = Groq(
            api_key=GROQ_API_KEY,
            http_client=httpx.Client(verify=False, follow_redirects=True),
        )

    def ask(self, question: str) -> str:
        """Send a question to the LLM and return the response text."""
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content
