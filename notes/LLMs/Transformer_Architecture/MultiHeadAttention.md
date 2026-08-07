 # Multi-Head Attention

Once we have our input embeddings and positional encodings, we're ready to pass this data into the **Transformer encoder block**, where it goes through a **Multi-Head Attention** layer followed by a **Feed-Forward Neural Network**.

Multi-Head Attention is one of the core components of the Transformer architecture. It allows the model to determine how important every token is relative to every other token in the input sequence.

The output of this process is a set of **attention vectors** that capture contextual relationships between all of the tokens in the text.

---

# Query, Key, and Value Vectors

For every token in the input sequence, the Transformer creates three vectors.

## Query (Q)

The **Query vector** represents the current token's question about the other tokens in the sequence.

You can think of it as asking:

> "Which other words are important for understanding me?"

The Query vector represents the token that is currently being processed.

---

## Key (K)

Each token also has a **Key vector**.

The Key vector contains information that helps determine whether another token should pay attention to it.

Each Query vector is compared against every Key vector in the sequence.

---

## Value (V)

Each token also has a **Value vector**.

The Value vector contains the actual information or meaning associated with that token.

Once the model determines which tokens deserve the most attention, the Value vectors provide the information that is passed forward.

---


# Step 1: Calculate Similarity Scores

The first step is to compare every Query vector with every Key vector.

This is done using the **dot product**.

The result is a similarity score for every token in the sequence.

Higher similarity scores indicate that two tokens are more closely related.

---

# Step 2: Scale the Similarity Scores

The similarity scores are divided by the square root of the Key vector dimension.

This scaling prevents the dot product values from becoming too large, helping maintain numerical stability during training.

---

# Step 3: Apply the Softmax Function

The scaled similarity scores are passed through the **Softmax** function.

Softmax converts the scores into probabilities that sum to 1.

These probabilities become the **attention weights**.

Higher attention weights indicate that the Query token should pay more attention to that particular token.

---

# Step 4: Compute the Weighted Sum

The attention weights are multiplied by the corresponding Value vectors.

The weighted Value vectors are then summed together.

The result is a new vector called the **attention vector**, which combines information from across the entire input sequence while emphasizing the most relevant tokens.

---

# Why Is It Called Multi-Head Attention?

Instead of performing this process only once, Transformers perform it multiple times simultaneously.

Each separate attention calculation is called an **attention head**.

Every head learns a different set of weights, allowing it to focus on different aspects or patterns within the input sequence.

For example:

- One head might learn grammatical relationships.
- One head might learn semantic relationships.
- One head might learn long-range dependencies.
- One head might learn contextual meaning.

The outputs from all attention heads are concatenated and passed through a final linear transformation to produce the output of the Multi-Head Attention layer.

---

# Advantages of Multi-Head Attention

- Allows the model to attend to multiple parts of the input simultaneously.
- Captures different types of relationships between words.
- Better understands long-range dependencies.
- Processes all tokens in parallel, making Transformers much faster than RNNs.

---

# Summary

Multi-Head Attention enables Transformers to understand relationships between every token in a sequence simultaneously.

The overall process is:

1. Create Query, Key, and Value vectors.
2. Compute similarity scores using dot products.
3. Scale the similarity scores.
4. Apply the Softmax function to obtain attention weights.
5. Compute a weighted sum of the Value vectors.
6. Repeat the process across multiple attention heads.
7. Concatenate the outputs to produce the final attention representation.