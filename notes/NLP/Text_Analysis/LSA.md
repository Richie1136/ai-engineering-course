# Latent Semantic Analysis (LSA)

## Overview

Latent Semantic Analysis (LSA) rests on two ideas.

### 1. Distributional Hypothesis

The Distributional Hypothesis states that words with similar meanings appear frequently together.

### 2. Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) recreates text documents into different vectors. Each vector expresses a different way of looking at meaning in the text.

The vectors can be expressed by the following equation:

* **M** - Document Term Matrix (rows are each individual document and columns are our terms).
* **U** - Document Topic Matrix (columns are our topics).
* **Sigma** - A vector whose values express how much each latent topic explains the variance in the overall data.
* **Vᵀ** - Terms Document Matrix (rows are our topics and columns are our terms). The small **T** means the matrix is transposed.

---

## Dimensionality Reduction

SVD is a method of dimensionality reduction.

The vectors can be used to identify similar words and documents by clustering and similarity scores.
