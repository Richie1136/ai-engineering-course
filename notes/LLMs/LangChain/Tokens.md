# Tokens

## Overview

Tokenization is the process of breaking text into smaller units called **tokens**.

These tokens are the pieces of text that a large language model (LLM) processes instead of entire words or sentences.

The collection of all possible tokens that a model understands is known as its **vocabulary**.

A useful rule of thumb is:

- **100 tokens ≈ 75 words**

This is because a token represents roughly three-quarters of a word on average.

---

## What Is a Token?

When discussing OpenAI's tokenization mechanism, a token is **not necessarily a single character or a complete word**.

Instead, a token is often a sequence of characters that commonly appear together.

For example:

- A complete word may be one token.
- A long word may be split into multiple tokens.
- A punctuation mark may be its own token.
- Frequently occurring character sequences may become single tokens.

---

## Why Do Language Models Use Tokens?

The goal of a language model is to learn the statistical relationships between the tokens in its vocabulary.

Once the model understands these relationships, it predicts the token that is most likely to appear next.

It then repeats this process one token at a time until it generates a complete response.

---

## Token Size Trade-Offs

The size of a token has a significant impact on how a language model performs.

### Smaller Tokens

Using smaller tokens reduces the vocabulary size, making the model more memory efficient.

Smaller tokens also make it easier for the model to learn:

- New words
- Different languages
- Rare vocabulary

However, smaller tokens make it more difficult for the model to capture the semantic meaning of text.

For example, predicting the next **character** is generally much harder than predicting the next **word**.

---

### Larger Tokens

Larger tokens provide more contextual information because they contain more text.

However, they greatly increase the vocabulary size.

Imagine defining an entire sentence as a single token.

Even a tiny change to that sentence would create a completely different token, making prediction much more difficult.

As a result, the ideal token size lies somewhere between individual characters and entire sentences.

---

## Tokenization Examples

Different OpenAI models can tokenize the same text differently.

For example:

### GPT-4 & GPT-3.5

- 60 tokens
- 257 characters

### GPT-5.x & O1/O3

- 57 tokens
- 257 characters

Although the text is identical, the newer tokenizer is able to represent it using fewer tokens.

This demonstrates that different models can have different vocabularies and tokenization strategies.

---

## Frequently Occurring Character Sequences

Tokenizers often group sequences of characters that frequently appear together into a single token.

Examples include:

- Common words
- Numbers
- Frequently occurring character combinations

For example:

```
1234567890
```

may be represented as fewer tokens than treating each digit individually.

This improves efficiency during both training and inference.

---

## Emoji Tokenization

Unicode characters, including emojis, are often split into multiple byte-based tokens.

For example:

```
👋🏾
```

may be represented by several individual tokens instead of one.

This is because emojis are encoded using multiple Unicode bytes rather than simple ASCII characters.

---

## Example: Beginning of a Sentence

Sentence:

> What would you like to have for dinner?

Both GPT-4/3.5 and GPT-5.x tokenize this sentence into:

- **9 tokens**
- **39 characters**

Notice that the word **"What"** appears as a single token.

Because it:

- Begins the sentence
- Starts with a capital letter
- Appears before a question mark

it receives its own unique token ID.

The question mark is also represented as its own token because it occurs very frequently.

---

## Example: Middle of a Sentence

Sentence:

> I don't mind what we have for dinner.

Although the word **"what"** is the same word, it receives a different token ID.

This is because:

- It appears in lowercase.
- It is preceded by a space.
- It appears in the middle of the sentence.

Tokenizers learn these patterns, so tokens often represent both the word and its surrounding context.

---

## Example: Inside a Title

Sentence:

> Have you listened to What a Wonderful World by Louis Armstrong?

Here the word **"What"** again receives a different token.

This is because:

- It is capitalized.
- It appears after a space.
- It is part of a title rather than the beginning of a sentence.

Once again, the tokenizer treats it as a different token because its surrounding context has changed.

---

## Key Takeaways

- Tokens are the basic units processed by language models.
- A token is usually a sequence of characters rather than a complete word.
- Approximately **100 tokens correspond to 75 words**.
- Language models predict text **one token at a time**.
- Smaller tokens improve flexibility but capture less meaning.
- Larger tokens capture more context but require much larger vocabularies.
- Different OpenAI models may tokenize the same text differently.
- Token IDs depend not only on the word itself but also on its surrounding context.
