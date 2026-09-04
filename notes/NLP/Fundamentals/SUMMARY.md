# NLP Fundamentals — Summary

## File Guide

- `IntroductionToNLP.md` — NLP goals, applications, and challenges.
- `TextPreprocessing.py` — Normalizes and cleans raw text.
- `Tokenization.py` — Divides text into sentences, words, or subword units.
- `PartsOfSpeech.py` — Assigns grammatical roles to tokens.
- `NamedEntityRecognition.py` — Extracts people, places, organizations, dates,
  and other real-world entities.

## Core Workflow

```text
raw text -> normalization -> tokens -> linguistic annotations
```

## Key Takeaways

- Preprocessing should match the downstream task; removing information blindly
  can reduce model quality.
- Tokenization defines the units that later analysis operates on.
- POS tagging describes grammatical function.
- NER converts important spans of unstructured text into labeled entities.

## Review Questions

1. When might punctuation or stop words carry useful meaning?
2. How is a POS tag different from an entity label?
3. Why must preprocessing be reproducible between training and inference?
