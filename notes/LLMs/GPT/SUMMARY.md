# GPT and the OpenAI API — Summary

## What This Section Covers

This section introduces GPT-style generative models and demonstrates basic
OpenAI API workflows.

## File Guide

- `GPTFundamentals.md` — Generative Pre-trained Transformer concepts.
- `OpenAIAPI.py` — Wraps a basic model request in a reusable function.
- `Chatbot.py` — Uses system instructions and few-shot conversation examples.
- `TextSummarization.py` — Transforms longer text into keywords or summaries.
- `config.py` — Loads the API key from the environment.

## Core Workflow

```text
instructions + input -> API client -> GPT model -> generated text
```

## Key Takeaways

- GPT generates text autoregressively, one token after another.
- System instructions establish behavior; user messages contain requests.
- Few-shot examples demonstrate desired behavior or output format.
- API keys belong in environment variables and must never be committed.
- Output length and model choice affect latency and cost.
