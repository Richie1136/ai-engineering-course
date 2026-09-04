# Text Representation — Summary

## What This Section Covers

Machine-learning algorithms require numbers, so text representation converts a
document collection into numeric feature vectors.

## File Guide

- `VectorizingText.md` — Why and how text becomes vectors.
- `BagOfWords.py` — Counts individual vocabulary terms.
- `NGrams.py` — Counts short sequences to preserve local word order.
- `TFIDF.py` — Weights terms by local frequency and corpus rarity.

## Quick Comparison

| Representation | Preserves | Main limitation |
| --- | --- | --- |
| Bag of Words | Word frequency | Discards word order |
| N-grams | Short local phrases | Rapidly increases feature count |
| TF-IDF | Relative term importance | Still lacks deep semantic context |

## Core Workflow

```text
documents -> fit vocabulary/statistics -> numeric document-term matrix
```

## Key Takeaway

The vectorizer must be fitted on training data and reused unchanged for new
text so every feature keeps the same meaning and position.
