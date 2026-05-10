# groq-llm-demo

A minimal Python project that calls an LLM via the [Groq API](https://console.groq.com/).

## Project Structure

```
groq-llm-demo/
├── app.py            # main entry point
├── config.py         # loads environment variables
├── llm/
│   ├── __init__.py
│   └── groq_client.py  # GroqLLM wrapper class
├── requirements.txt
└── README.md
```

## Setup

1. **Clone / enter the project directory**

   ```bash
   cd groq-llm-demo
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Groq API key**

   Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_api_key_here
   ```

   Get a free key at <https://console.groq.com/keys>.

## Run

```bash
python app.py
```

The model (`llama3-8b-8192`) will analyze the sample idea and print its response.
