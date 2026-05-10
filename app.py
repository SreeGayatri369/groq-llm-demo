from llm.groq_client import GroqLLM

QUESTION = (
    "I want to know whether my idea will work or not. "
    "My idea is to build a mobile app that helps people find local farmers markets. "
    "Please analyze and tell me."
)


def main():
    llm = GroqLLM()
    print("Question:", QUESTION)
    print("\nAnswer:", llm.ask(QUESTION))


if __name__ == "__main__":
    main()
