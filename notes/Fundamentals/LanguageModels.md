# Language Models

> **Key takeaway:** Language modeling evolved by increasing how much context a
> model can use and how efficiently it can learn relationships within that
> context.

## Evolution of Language Models

```text
N-Grams
   ↓
RNNs (Recurrent Neural Networks)
   ↓
LSTMs (Long Short Term Memory)
   ↓
Transformers
   ↓
Modern LLMs (GPT, Claude, Gemini)
```


## What Is a Language Model?

Language models are AI systems that learn patterns in language and predict words or tokens based on context.

## Four Core Language-Modeling Techniques

### 1. N-Grams

N-grams are one of the earliest language modeling techniques. They estimate the probability of a word based on the previous n-1 words.

Types:

- Unigram (n=1)
  - Uses no context.
  - Predicts based on overall word frequency.

- Bigram (n=2)
  - Uses one previous word.

- Trigram (n=3)
  - Uses two previous words.

Example:

Sentence: "My favorite sport is _____"

Unigram:
- Looks for the most common word in the training data.

Bigram:
- Looks at words that commonly follow "is".

Trigram:
- Looks at words that commonly follow "sport is".

Limitation:
N-grams struggle with long-range context because they only consider a limited number of previous words.

--------------------------------------------------

### 2. Recurrent Neural Networks (RNNs)

RNNs were designed to process sequential data such as text.

Advantages:
- Can remember previous words.
- Capture word order and context better than N-grams.

Example:

"The dog chased the ball because it was..."

The RNN can use previous words to help predict the next word.

Limitation:
- Vanishing Gradient Problem
- Struggle with long documents because earlier information is gradually forgotten.

--------------------------------------------------

### 3. Long Short-Term Memory Networks (LSTMs)

LSTMs improve upon RNNs by introducing gates that control which information should be remembered and which should be forgotten.

Advantages:
- Better long-term memory.
- Improved handling of long text sequences.

Key Concept:
- Gate Architecture
- Long-Term Memory Retention

Limitation:
- Computationally expensive.
- Slow to train on very large datasets.

--------------------------------------------------

### 4. Transformers

Transformers are the foundation of modern AI systems such as ChatGPT, Claude, Gemini, and Llama.

Transformers use an Attention Mechanism that allows the model to focus on the most important words in a sequence.

Attention Mechanism:
- Assigns attention scores to words.
- Important words receive higher attention.
- Allows the model to understand long-range dependencies.

Advantages:
- Scalable
- Parallel processing
- Handles long contexts
- More efficient than RNNs and LSTMs

Examples:
- GPT
- Claude
- Gemini

## Quick Comparison

| Technique | Context mechanism | Main limitation |
| --- | --- | --- |
| N-gram | Previous `n - 1` tokens | Very limited context |
| RNN | Recurrent hidden state | Earlier information fades |
| LSTM | Gated recurrent memory | Sequential and expensive to train |
| Transformer | Attention across tokens | Compute and memory requirements |
