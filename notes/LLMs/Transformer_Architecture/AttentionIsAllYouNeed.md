# Attention Is All You Need

## What is Attention?

The attention mechanism enables a model to weigh the importance of different words or tokens in an input sequence when producing an output.

Instead of processing the entire input sequence in a fixed manner, the model can selectively focus on the most relevant parts of the input at each step of processing. This is accomplished by assigning an **attention score** (or weight) to every token in the input sequence.

Tokens with higher attention scores are considered more important for the current prediction. These scores are then used to compute a weighted representation of the input, which helps generate the output.

This mechanism allows Transformers to capture long-range dependencies between words, even when they are far apart in a sentence. In simple terms, attention is a weighting mechanism that enables the model to focus on the most relevant parts of the input when producing an output.

---

## Why is Attention Needed?

The paper demonstrates the importance of attention using a language translation example.

Suppose we want to translate the following French sentence into English.

**French:**

"L'accord sur la zone economique europeenne a ete signe aout 1992"

**English:**

"The agreement on the European Economic Area was signed in August 1992."

A simple translation model could translate each word individually, one token at a time. However, this approach would produce an incorrect translation because French and English often have different word orders.

For example, translating word by word might produce:

> Area Economic European

instead of the correct English phrase:

> European Economic Area

Because of differences in grammar and sentence structure, a model needs to consider surrounding words rather than translating each word independently.

Attention makes this possible by allowing the model to look at other relevant words in the sentence before deciding how each word should be translated.

---

## Self-Attention

When studying Transformers, you'll frequently encounter the term **self-attention**.

Self-attention is a specific type of attention mechanism that computes relationships between words within the same input sequence.

Rather than only looking at nearby words, self-attention allows every token to consider every other token in the sentence. This enables the model to capture context, understand relationships between words, and learn long-range dependencies much more effectively than previous sequence models.

Self-attention is one of the core innovations that makes the Transformer architecture so powerful.

---

## Key Takeaways

- The attention mechanism allows a model to determine which words or tokens are most important when generating an output.
- Each token is assigned an **attention score**, indicating how much focus the model should place on it.
- Attention enables Transformers to capture **long-range dependencies**, allowing them to understand relationships between words that are far apart in a sentence.
- Word-by-word translation is often incorrect because different languages have different grammatical structures and word orders.
- Attention improves translation by allowing the model to consider the entire context of a sentence rather than treating each word independently.
- **Self-attention** is a special type of attention that allows every token in an input sequence to attend to every other token in the same sequence.
- Self-attention enables Transformers to build rich contextual representations of text and is one of the key innovations behind the Transformer architecture.