# BERT — Summary

## What This Section Covers

BERT is a bidirectional Transformer encoder designed to build contextual text
representations. These notes focus on extractive question answering.

## File Guide

- `BERTArchitecture.md` — Encoder architecture and bidirectional context.
- `GPTvsBERT.md` — Generative decoder models versus encoder models.
- `LoadingModelAndTokenizer.py` — Loads a matching QA model/tokenizer and
  inspects answer scores.
- `CalculatingResponse.py` — Explains start-token and end-token prediction.
- `QABot.py` — Builds an FAQ-style extractive question-answering prototype.
- `BERTvsRoBERTavsDistilBERT.py` — Compares related encoder model families.

## Core Workflow

```text
question + context -> tokenizer -> BERT -> start/end scores
                   -> selected token span -> answer
```

## Key Takeaways

- BERT reads context bidirectionally rather than generating left to right.
- Extractive QA selects an answer already present in the supplied passage.
- Model and tokenizer checkpoints must match.
- RoBERTa changes BERT's training recipe; DistilBERT reduces size and latency.
