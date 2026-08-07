# =============================================================================
# The Problem with Recurrent Neural Networks (RNNs)
# =============================================================================

Different types of neural networks are designed to solve different kinds of
problems. For example, Convolutional Neural Networks (CNNs) are designed to
process visual information and excel at image classification tasks.

While neural networks perform extremely well with images, language presents a
much more difficult challenge because words depend heavily on context and
their position within a sentence.

# =============================================================================
# Recurrent Neural Networks (RNNs)
# =============================================================================

Before Transformers, Recurrent Neural Networks (RNNs) were one of the most
popular architectures for Natural Language Processing (NLP).

Suppose we want to translate a sentence from English to French.

An RNN processes the sentence one word at a time, preserving the order of the
words before producing the translated output.

This sequential processing is important because changing the order of words
can completely change the meaning of a sentence.

# =============================================================================
# The Main Problem with RNNs
# =============================================================================

Although RNNs preserve word order, they struggle when processing long pieces
of text.

As the sequence becomes longer, the network gradually loses information from
the beginning of the text.

By the time an RNN reaches the end of a long document, it may have forgotten
important context that appeared much earlier.

This makes it difficult to understand long articles, essays, conversations,
or books.

For language models to work effectively, they need some way of remembering
important information that appeared earlier in the text.

# =============================================================================
# Example
# =============================================================================

"The New York Times is a daily newspaper. It was first issued in 1851."

Humans immediately understand that **"It"** refers to **"The New York Times."**

An RNN, however, may struggle to maintain this relationship if the two words
are separated by a large amount of text.

Without remembering the earlier context, the true meaning of the sentence can
be lost or misunderstood.

# =============================================================================
# Another Limitation: Sequential Processing
# =============================================================================

Another major limitation of RNNs is that they process one word after another.

Because every word depends on the previous word being processed first, RNNs
cannot efficiently process multiple words in parallel.

As a result:

- Training is much slower.
- Large datasets become difficult to process.
- Scaling to billions of words becomes impractical.

This limitation made RNNs unsuitable for training the extremely large language
models we use today.

# =============================================================================
# The Transformer Solution
# =============================================================================

Transformers solve many of the limitations of recurrent neural networks.

Unlike RNNs, Transformers can process many words simultaneously through
parallelization, making training significantly faster.

More importantly, Transformers use a mechanism called **Attention**.

Attention allows the model to focus on the most important words in a sentence,
regardless of how far apart those words are.

Instead of forgetting information from earlier in the text, the model can
directly reference relevant words whenever they are needed.

This gives Transformers a much stronger understanding of context than
traditional recurrent neural networks.

# =============================================================================
# Key Takeaways
# =============================================================================

- RNNs process text one word at a time.
- They preserve word order but struggle with long-term context.
- RNNs cannot efficiently parallelize computations.
- These limitations make training very large language models impractical.
- Transformers solve these problems through parallelization and the Attention
  mechanism.
- Modern Large Language Models (LLMs), such as ChatGPT, are built on the
  Transformer architecture.