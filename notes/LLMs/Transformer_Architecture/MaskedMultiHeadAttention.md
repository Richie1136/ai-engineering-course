# Masked Multi-Head Attention

The desired outputs that we want the model to learn are fed into the **decoder block**.

Think back to our example of translating French to English. The French words are fed into the **encoder block**, while the English words are fed into the **decoder block**.

These outputs go through the same embedding process and receive their positional encodings before being passed to the decoder block.

However, we do **not** feed all of the output embeddings into the decoder block at once.

---

# What Is Masked Multi-Head Attention?

Instead, the output embeddings first pass through the **Masked Multi-Head Attention** layer.

This layer is similar to the Multi-Head Attention layer in the encoder block, but it is called **masked** because some of the output information is hidden from the model during training.

The model is allowed to see all of the French input words. When we refer to the words, we are really referring to the attention vectors that represent those words.

However, the model can only see the English words that come **before** the current word being processed in the output sequence.

---

# Why Is It Masked?

The model must learn to predict the next word in the sequence instead of simply looking ahead at the correct answer.

Words that appear later in the output sequence are **masked**, meaning they are hidden from the model during this stage.

This forces the model to learn the correct next word based only on the words that have already been generated.

---

# Summary

Masked Multi-Head Attention is used in the **decoder block** of the Transformer architecture.

Its purpose is to:

- Hide future output words from the model.
- Allow the model to see only previously generated output words.
- Force the model to predict the next word in the sequence instead of looking ahead at the correct answer.