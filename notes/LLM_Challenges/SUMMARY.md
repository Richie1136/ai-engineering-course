# LLM Challenges — Summary

## What This Section Covers

This section examines practical constraints that affect the quality,
responsiveness, and future development of language models.

## File Guide

- `Inconsistency.md` — Why the same prompt can produce varying responses.
- `Latency.md` — Why generation can be slow and why response time matters.
- `RunningOutOfData.md` — Limits on high-quality training data.

## Key Takeaways

- Model output is probabilistic, so consistency must be designed and tested.
- Latency depends on model size, prompt length, output length, infrastructure,
  and application architecture.
- More data is not automatically better; quality, diversity, legality, and
  provenance matter.
- Production systems manage these limits through model selection, caching,
  prompt design, retrieval, monitoring, and evaluation.
