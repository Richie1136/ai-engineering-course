# Predicting Final Outputs

The Multi-Head Attention layer in the decoder block receives inputs from both the **encoder block** and the **Masked Multi-Head Attention** layer.

In addition to self-attention, the decoder's Multi-Head Attention layer calculates attention scores between the current output token and the encoder outputs.

This step helps the model determine which parts of the input sequence are most relevant when generating the next token in the output sequence.

---

# Creating the Context Vector

The encoder outputs are weighted using these attention scores to create a **context vector**.

The context vector represents the relevant information from the input sequence that should be considered when generating the current output token.

---

# Feed-Forward Layer

The Feed-Forward Layer processes the output from the attention mechanism, making it more suitable for the next stage of the decoder.

---

# Linear Layer and Softmax

The **Linear Layer** is another feed-forward neural network that transforms the output into a useful format.

The **Softmax** layer then converts the output into a probability distribution.

This probability distribution is used to predict the next word in the output sequence.

---

# Repeating the Process

The entire decoder process is repeated for every token in the output sequence.

At each step, the previously generated output tokens are included as input when predicting the next token.

---

# Summary

The decoder predicts the final output by:

1. Receiving information from both the encoder block and the Masked Multi-Head Attention layer.
2. Calculating attention scores between the current output token and the encoder outputs.
3. Creating a context vector from the weighted encoder outputs.
4. Passing the result through a Feed-Forward Layer.
5. Using a Linear Layer and Softmax to predict the next word.
6. Repeating the process for every token in the output sequence.