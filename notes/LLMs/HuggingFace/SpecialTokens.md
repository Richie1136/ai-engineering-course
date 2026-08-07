# Special Tokens

## Overview

Special tokens are a fundamental concept in the world of large language models.

Special tokens are specific placeholders or markers that help a model perform various tasks and handle specific instructions within text.

They act like signposts that help the model understand the structure and context of the input while guiding its behavior for different tasks.

Using the correct special tokens ensures that the output of the tokenization process is in a format that the chosen model understands.

---

## Types of Special Tokens

### CLS and SEP Tokens

**CLS** stands for **Classification**, and **SEP** stands for **Separator**.

These special tokens are commonly used in tasks such as:

- Text classification
- Sentence pair classification

The **CLS** token is typically placed at the beginning of the input sequence.

The **SEP** token is used to separate different segments of text.

In a typical classification task, you might provide one sentence or two related sentences, and the model uses these special tokens to understand the structure of the input.

---

### MASK Token

The **MASK** token is used in tasks related to masked language modeling or text generation where a word has been intentionally removed.

For example:

```text
Fine tuning is [MASK] for all!
```

The model's task is to predict the missing word.

This type of training is commonly used for tasks such as text completion.

---

### Task-Specific Tokens

Some tasks require custom special tokens to guide the model's behavior.

For example, translation models may use tokens such as:

- `[SOURCE]`
- `[TARGET]`

These tokens indicate the source language and target language, helping guide the translation process.

---

### Padding and Truncation Tokens

When multiple sentences of different lengths are passed into a model, additional preprocessing is often required.

One approach is **padding**, where extra padding tokens are added so that every input has the same length.

For example:

```text
1. Fine tuning is fun for all [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

2. Fine tuning a large language model involves refining its parameters and adjusting its training data to specialize its understanding or performance in a particular domain, task, or language.

3. Fine tuning is refining parameters to specialize understanding and performance in specific tasks [PAD] [PAD] [PAD] [PAD]
```

Another approach is **truncation**, where longer inputs are shortened to a specified length.

For example:

```text
1. Fine tuning is fun for all

2. Fine tuning a large language model involves refining its parameters

3. Fine tuning is refining parameters to specialize understanding and performance
```

Padding and truncation ensure that the inputs are consistent and match the requirements of the model's architecture.

Each language model has its own requirements for special tokens. This is why it is important to use the tokenizer that matches the model you are working with. The tokenizer automatically formats the input using the appropriate special tokens for that model.