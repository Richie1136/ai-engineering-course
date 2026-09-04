# XLNet — Summary

## What This Section Covers

XLNet combines autoregressive prediction with permutation-based training so it
can learn bidirectional context without BERT's masked-token mismatch.

## File Guide

- `GPTvsBERTvsXLNET.md` — Compares the training objectives and strengths of
  three Transformer model families.
- `XLNetPreprocessing.py` — Cleans data, encodes labels, tokenizes text, creates
  datasets, trains/evaluates a classifier, and performs inference.
- `fine_tuned_model/` — Saved model artifacts produced by training.
- `test_trainer/` — Trainer checkpoints and optimizer state.

## Training Workflow

```text
raw labeled text -> clean/split -> XLNet tokenizer -> datasets
                 -> fine-tune -> evaluate -> save -> inference
```

## Key Takeaways

- XLNet predicts tokens across different factorization orders.
- Classification requires labels encoded as numeric class IDs.
- Training, validation, and test data serve different purposes.
- Saved checkpoints are generated artifacts, not lesson source files.
