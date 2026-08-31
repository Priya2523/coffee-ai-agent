# Bean Buddy — AI Coffee Ordering Agent

Bean Buddy is a customer-facing AI agent that recommends coffee (and other menu items) based on what a customer is craving right now, blended with their past orders. Built for **Track 1 of the H2S GenAI Academy APAC Edition** — *"Build and Deploy a Customer-Facing AI Agent."*

- **Live app:** https://coffee-ai-agent.onrender.com
- **Repo:** https://github.com/Priya2523/coffee-ai-agent

> ⏳ The live app runs on Render's free tier, which sleeps after inactivity. The first request after a period of inactivity can take **~30–50 seconds** to wake the service — please be patient on the first message.

## About this build vs. the official lab spec

The official Track 1 spec asks for **Google ADK (Agent Development Kit)** deployed to **Cloud Run**. I don't have a GCP account, so this project substitutes equivalent, GCP-free building blocks that preserve the same agent pattern — retrieve relevant context, ground the LLM in it, respond to the customer.

| Concern | Official spec | This project |
|---|---|---|
| Agent framework | Google ADK | Plain Python function (`ask_agent`) implementing the same retrieve → prompt → generate loop |
| Retrieval / RAG | ADK's built-in RAG tooling | scikit-learn `TfidfVectorizer` + cosine similarity over the menu |
| LLM | Gemini (via ADK) | Groq `openai/gpt-oss-120b` |
| Hosting | Cloud Run | Render (free web service, `render.yaml`) |
| UI | Not specified / custom | Gradio `ChatInterface` |

The substitutions are functionally equivalent for this use case — the agent still retrieves grounded context before generating a response — just implemented without GCP-specific tooling.

## How it works

1. The customer sends a message describing their mood or craving (e.g. "something sweet and cold").
2. Their simulated order history is pulled from `CUSTOMER_HISTORY` and merged into the retrieval query.
3. `TfidfVectorizer` + cosine similarity rank the menu (`MENU` list) against that query, returning the top matching items.
4. Only those retrieved items are passed to Groq's `openai/gpt-oss-120b` model, along with a system prompt instructing it to recommend **only** from what was retrieved (no invented items or prices).
5. The model replies conversationally and always ends with `Top pick: <item name> - ₹<price>`.
6. Everything is served through a Gradio `ChatInterface` web UI.

### Architecture

```
 Customer query + mood/craving
            │
            ▼
 ┌─────────────────────────┐
 │  Order history lookup    │  (CUSTOMER_HISTORY dict)
 └───────────┬──────────────┘
             ▼
 ┌─────────────────────────┐
 │  TF-IDF retrieval         │  scikit-learn: TfidfVectorizer
 │  (cosine similarity)      │  ranks MENU items vs. query + history
 └───────────┬──────────────┘
             ▼  top-k relevant menu items
 ┌─────────────────────────┐
 │  Groq LLM                 │  openai/gpt-oss-120b
 │  (grounded generation)    │  system prompt: recommend ONLY retrieved items
 └───────────┬──────────────┘
             ▼  natural-language reply + "Top pick: <item> - ₹<price>"
 ┌─────────────────────────┐
 │  Gradio ChatInterface UI  │
 └─────────────────────────┘
```

## Files

- **`app.py`** — the agent logic (menu, retrieval, Groq call) and Gradio UI.
- **`requirements.txt`** — Python dependencies (`gradio`, `scikit-learn`, `groq`).
- **`render.yaml`** — Render deployment config (build/start commands, env vars).

## Running locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_groq_api_key_here   # Windows: set GROQ_API_KEY=your_key
python app.py
```

The app starts a Gradio server (default port `7860`, or `$PORT` if set).

## Deploying to Render

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. On [Render](https://render.com), click **New → Blueprint** (or **New → Web Service**) and connect this GitHub repo.
3. Render auto-detects `render.yaml` and configures the build (`pip install -r requirements.txt`) and start (`python app.py`) commands.
4. Add an environment variable **`GROQ_API_KEY`** with your Groq API key (marked `sync: false` in `render.yaml`, so Render will prompt you to enter it rather than committing it to the repo).
5. Deploy. Note Render's free-tier services sleep after inactivity and take ~30–50 seconds to wake on the next request.
