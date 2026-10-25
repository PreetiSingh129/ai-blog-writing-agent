# AI Blog Writing Agent

AI Blog Writing Agent is a Streamlit app that creates blog posts with Google Gemini and improves them through an automated review-and-revision workflow built with LangGraph.

## Features

- Generate a blog from a topic and optional instructions.
- Choose the target audience, writing tone, and approximate length.
- Configure separate Gemini models for writing and reviewing.
- Have a reviewer check clarity, structure, language, and missing points.
- Automatically revise the draft up to a configurable limit.
- View reviewer feedback, revision history, workflow status, and execution log.
- Download the final post as Markdown or plain text.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key

## Setup

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/PreetiSingh129/ai-blog-writing-agent.git
   cd ai-blog-writing-agent
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Gemini API key:

   ```dotenv
   GEMINI_API_KEY=your_gemini_api_key
   ```

   Keep this key private. The `.env` file is excluded from Git.

## Run the web app

Start Streamlit from the project directory:

```bash
streamlit run ui.py
```

Open the local URL printed by Streamlit. Enter a topic, optionally add instructions, adjust the settings in the sidebar, and select **Generate Blog**. The configured Gemini model IDs must be available to your API key.

## Run the command-line example

The `app.py` script runs a simple LangGraph workflow for a sample topic and prints the generated blog and review in the terminal:

```bash
python app.py
```

## Project files

- `ui.py` — Streamlit interface and configurable draft/review/revision workflow.
- `app.py` — Minimal command-line LangGraph example.
- `check_models.py` — Lists Gemini models available to the configured API key.
- `requirements.txt` — Python dependencies.

## Security

Never commit your Gemini API key. Store it only in your local `.env` file or in your deployment platform's secret manager. Do not share generated content containing private or confidential information with an external model unless you are authorized to do so.
