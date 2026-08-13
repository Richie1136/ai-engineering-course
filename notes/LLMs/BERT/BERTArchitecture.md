# BERT Architecture

## Overview

While BERT is based on the transformer architecture discussed earlier in the course, it introduces several modifications and uses a specific pre-training methodology that makes it effective for a wide range of natural language processing (NLP) tasks.

---

## Encoder-Only Architecture

The original transformer architecture contains both an **encoder** block and a **decoder** block that process the inputs and outputs during training.

One of the main differences in BERT's architecture is that it only uses the **encoder** portion of the transformer.

This is because BERT is primarily designed for pre-training on large text corpora and understanding text, rather than performing sequence-to-sequence tasks.

Other transformer-based models, such as the GPT family of models, are built around the **decoder** because they are designed for text generation and sequence-to-sequence tasks.

BERT, in contrast, focuses on encoding and understanding text, and it has proven to be highly effective in this role.

---

## Encoder Layers

The BERT architecture consists of a stack of identical encoder layers.

The number of encoder layers depends on the specific version of BERT:

- **BERT Base** – 12 encoder layers
- **BERT Large** – 24 encoder layers

The deeper architecture of BERT Large allows the model to capture more complex patterns and relationships between tokens.

---

## Input Embeddings

BERT accepts tokenized text as its input.

Each token is converted into an embedding vector using three different types of embeddings.

### Token Embeddings

Token embeddings are the standard word embeddings that convert each token into a meaningful numerical representation.

Special tokens such as **[CLS]** and **[SEP]** are also added to the input sequence.

### Segment Embeddings

BERT can accept pairs of text sequences as input.

Segment embeddings help the model distinguish between the different text segments.

### Positional Embeddings

Positional embeddings encode the position of each token within the input sequence.

This allows the model to understand the order of the words.

---

## BERT Pre-Training Objectives

BERT's pre-training is focused on understanding language bidirectionally.

The model is trained using two primary objectives.

### Masked Language Modeling

In Masked Language Modeling (MLM), a percentage of the tokens in each input sequence are randomly replaced with a **[MASK]** token.

The model is then trained to predict the missing word using the surrounding context.

The goal is to determine which word should replace each masked token.

This forces the model to learn bidirectional context because it can see both the words before and after the masked token.

### Next Sentence Prediction

The second objective is **Next Sentence Prediction (NSP)**.

BERT is trained to determine whether two sentences originally appeared next to each other.

During training:

- 50% of the sentence pairs are consecutive sentences from the original text.
- The remaining 50% are paired with a random sentence.

The goal is for the model to predict whether the second sentence actually follows the first.

This objective helps BERT learn the relationships between sentences.

Because these pre-training tasks focus on understanding language rather than generating text, a decoder is not required.

---

## Fine-Tuning

After pre-training, BERT can be fine-tuned for specific downstream NLP tasks by adding a task-specific output layer on top of the pre-trained encoder.

The output layer is designed specifically for the task being performed.

For example, if we wanted to perform sentiment analysis, the final output layer would convert the encoder's output into sentiment labels.

This flexibility and adaptability are two of BERT's biggest advantages.