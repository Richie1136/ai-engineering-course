# GPT vs BERT vs XLNet

## Overview

XLNet is a large language model developed by Google AI in collaboration with universities and other institutions in 2019.

It comes in two sizes:

- **XLNet Base** – 110 million parameters
- **XLNet Large** – 340 million parameters

XLNet was designed to overcome some of the limitations of previous language models by developing a deeper understanding of context during training.

---

## XLNet Architecture

Unlike BERT, which consists of encoder layers stacked on top of one another, XLNet is a **decoder-only** architecture.

Several decoder layers are stacked together to create the model:

- **XLNet Base** – 12 decoder layers
- **XLNet Large** – 24 decoder layers

One of the major differences between XLNet and earlier models, such as BERT, is the way it processes context during training.

---

## GPT vs BERT vs XLNet

Traditional autoregressive language models such as **GPT** are trained to predict the next word in a sequence given the previous words.

This prediction typically happens from **left to right**.

In contrast, **BERT** uses a **Masked Language Modeling (MLM)** objective.

Some words in a sentence are randomly masked, and the model predicts the missing words using the surrounding context.

XLNet takes a different approach during pre-training.

---

## Permutation-Based Training

During pre-training, XLNet can consider the entire context of a sentence instead of only looking at the words before or after the target word.

XLNet samples different **permutations** of the input sequence.

This means it considers different orders for predicting words throughout the sentence.

By doing this, the model learns to predict the probability of each word using the context it has learned from the entire sequence.

This results in improved contextual understanding.

---

## Bidirectional Context

One of the biggest advantages of permutation-based training is that XLNet can effectively capture bidirectional context.

During training, it is as if the model predicts each word by considering all of the other words in the sequence, rather than only the words immediately before or after it.

This helps the model understand relationships between words in a more comprehensive manner.

---

## Independence Assumption

XLNet makes an independence assumption when computing the probability of a word.

It assumes that the probability of generating a word is conditionally independent of the other words in the sequence, given the entire context.

To put it simply, the model does not rely only on word order when predicting a word.

Instead, it considers the context of the entire sequence.

---

## Advantages of XLNet

The permutation-based training used by XLNet overcomes several limitations of purely autoregressive models such as GPT.

While autoregressive models generate words sequentially, they may struggle to capture long-range dependencies and bidirectional context effectively.

XLNet still predicts words one at a time, but it considers the entire sequence during training, allowing it to develop a stronger understanding of context.

---

## Performance

XLNet achieved state-of-the-art results on a variety of benchmark NLP tasks, including:

- Question answering
- Text classification
- Language modeling

Its ability to model bidirectional context effectively and capture long-range dependencies contributed significantly to its success.