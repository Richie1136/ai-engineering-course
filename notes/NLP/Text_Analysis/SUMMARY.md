# Text Analysis — Summary

## What This Section Covers

This section uses sentiment analysis to measure expressed attitudes and topic
modeling to discover recurring themes across a document collection.

## File Guide

- `SentimentAnalysis.md` — Sentiment concepts and use cases.
- `RuleBasedSentiment.py` — TextBlob and VADER lexical scoring.
- `TransformerSentiment.py` — Contextual sentiment inference with a pretrained
  Transformer pipeline.
- `PracticalTaskSentiment.py` — Applied sentiment-analysis exercise.
- `TopicModeling.md` — Unsupervised theme-discovery concepts.
- `LDA.md` / `LDA.py` — Probabilistic document/topic distributions.
- `LSA.md` / `LSA.py` — Matrix-factorization-based latent concepts.

## Quick Comparison

| Method | Strength | Limitation |
| --- | --- | --- |
| Rule-based sentiment | Fast and interpretable | Limited contextual understanding |
| Transformer sentiment | Learns contextual patterns | Higher compute and lower transparency |
| LDA | Probabilistic, interpretable topics | Requires preprocessing and topic choice |
| LSA | Efficient latent semantic structure | Components may be harder to label |

## Key Takeaway

Analysis quality depends on preprocessing, method assumptions, evaluation, and
whether the discovered score or topic answers the real question.
