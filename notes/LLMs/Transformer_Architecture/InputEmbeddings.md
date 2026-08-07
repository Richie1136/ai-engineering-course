# Input Embeddings

The first step in a Transformer architecture is creating the input embeddings. If you're familiar with natural language processing (NLP), you'll know that we can't feed raw text directly into a model. We first need to convert it into a numerical representation.

## Creating Input Embeddings

### Step 1: Break the Input into Tokens

Tokens can be words, subwords, or even individual characters, depending on the tokenization strategy being used.

For example, the sentence:

> "I love natural language processing"

might be tokenized into:

```text
["I", "love", "natural", "language", "processing"]
```

### Step 2: Convert Tokens to Token IDs

Each token is mapped to a unique identifying number (called a **token ID**) based on a predefined vocabulary.

This vocabulary is typically built from a large corpus of text and contains all of the tokens that the model can recognize.

For example:

| Token | Token ID |
|-------|---------:|
| I | 34 |
| love | 56 |
| natural | 782 |
| language | 913 |
| processing | 4210 |

### Step 3: Retrieve Word Embeddings

Once each token has been converted into its corresponding token ID, the model retrieves a pre-trained embedding from an **embedding matrix**.

An embedding matrix contains a vector representation for every token in the vocabulary.

For example:

```text
I → [0.25, -0.12, 0.88, ...]
```

These vectors encode semantic and syntactic information about words. Words with similar meanings tend to have embedding vectors that are located close together in vector space.

## Positional Encoding

Word embeddings alone do not tell the model where a word appears within a sentence.

To solve this problem, Transformers use **positional encoding**, which provides information about each token's position in the input sequence.

Unlike recurrent neural networks (RNNs), Transformers process all tokens simultaneously. Positional encoding allows the model to understand word order without processing one word at a time.

## Padding and Truncation

Transformers require input sequences to have the same length.

To accomplish this, one of two techniques is commonly used:

### Padding

Padding adds special padding tokens (or zeros) to shorter sequences so that every input has the same length.

### Truncation

Truncation removes tokens from sequences that exceed the model's maximum input length.

We'll explore both of these techniques later in the course when working through practical examples.

## Summary

The embedding process converts text into numerical vectors that capture both semantic meaning and positional information.

These input embeddings are then passed into the encoder block of the Transformer, allowing the model to understand relationships between words and process natural language effectively.