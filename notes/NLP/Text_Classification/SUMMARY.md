# Text Classification — Summary

## What This Section Covers

Text classification assigns predefined labels to documents using numerical
text features and supervised machine-learning algorithms.

## File Guide

- `TextClassification.py` — End-to-end classification pipeline.
- `NaiveBayes.py` — Probabilistic baseline suited to count features.
- `LogisticRegression.py` — Linear classification with Bag of Words and TF-IDF.
- `LinearSupportVectorMachine.py` — Maximum-margin linear classification.
- `FakeNewsCaseStudy.py` — Full workflow combining exploration,
  preprocessing, analysis, topic modeling, classification, and evaluation.

## Standard Workflow

```text
labeled text -> train/test split -> vectorizer fit on training text
             -> classifier training -> unseen-text predictions -> metrics
```

## Key Takeaways

- Never fit the vectorizer on test data; that leaks information.
- Accuracy alone can hide class-specific failures, so inspect precision,
  recall, F1 scores, and a confusion matrix when appropriate.
- A simple baseline is necessary before judging whether complexity helps.
- Evaluation data must represent the conditions expected after deployment.
