# Transformer Architecture — Summary

## What This Section Covers

This section follows information through a Transformer, from token embeddings
to attention, feed-forward processing, and final token predictions.

## File Guide

- `DeepLearningRecap.md` — Neural-network concepts needed for Transformers.
- `ProblemWithRNNs.md` — Sequential processing and long-range memory limits.
- `AttentionIsAllYouNeed.md` — The paper and central attention idea.
- `TransformerArchitecture.md` — High-level encoder/decoder structure.
- `InputEmbeddings.md` — Numerical token and position representations.
- `MultiHeadAttention.md` — Parallel attention relationships.
- `MaskedMultiHeadAttention.md` — Preventing access to future tokens.
- `FeedForwardLayer.md` — Per-position nonlinear transformation.
- `PredictingFinalOutputs.md` — Converting hidden states into token scores.

## Core Flow

```text
tokens -> embeddings + positions -> attention -> feed-forward layers
       -> vocabulary scores -> next-token probabilities
```

## Key Takeaway

Attention lets every position selectively combine information from other
positions. Multi-head attention learns several relationship types, while
masking preserves autoregressive generation behavior.
