# Transformer Architecture

Before we start using Transformers, let's think back to our example of translating between French and English.

If we chose to use an RNN (Recurrent Neural Network) for this task, each word would be passed into the model one at a time. This sequential processing makes it difficult to capture long-range relationships and limits how efficiently the model can be trained.

With a Transformer, however, all of the input words can be fed into the model simultaneously rather than one at a time. This parallel processing is one of the biggest advantages of the Transformer architecture, making it much faster and better at understanding relationships between words throughout an entire sentence.

How is this possible?

The Transformer uses an **encoder-decoder architecture**, which we'll explore in more detail throughout this section.