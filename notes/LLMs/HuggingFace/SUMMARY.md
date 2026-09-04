# Hugging Face — Summary

## What This Section Covers

Hugging Face provides a model hub and libraries for discovering, loading,
running, fine-tuning, saving, and sharing pretrained machine-learning models.

## File Guide

- `HuggingFace.md` — Ecosystem and model-hub overview.
- `TransformerPipeline.py` — High-level inference with `pipeline()`.
- `PretrainedTokenizers.py` — Direct tokenization and encoded model inputs.
- `SpecialTokens.md` — Padding, classification, separation, and unknown tokens.
- `HuggingFacePyTorchTensorFlow.py` — Direct framework-level model inference.
- `SavingLoadingModels.py` — Persists compatible models and tokenizers.

## Core Workflow

```text
text -> matching tokenizer -> tensors -> pretrained model -> task output
```

## Key Takeaways

- Pipelines are convenient; direct model calls provide greater control.
- A tokenizer must match the model checkpoint.
- Special tokens communicate structure required by a model architecture.
- Saving both model and tokenizer makes later inference reproducible.
