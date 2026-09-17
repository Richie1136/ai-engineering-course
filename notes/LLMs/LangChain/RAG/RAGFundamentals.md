# RAG Fundamentals

## What are the limitations of LLMs?

Large Language Models (LLMs) are state-of-the-art models trained on vast amounts of data and can answer questions on a wide variety of topics.

Although they continue to improve rapidly, they have several limitations:

- Their knowledge has a training cutoff date.
- They cannot answer questions about proprietary or private company data out of the box.

Imagine combining the reasoning ability of an LLM with your own custom data.

This would allow an LLM to:

- Analyze documents
- Summarize reports
- Tag files
- Answer questions about company knowledge
- And much more

---

# Three Ways to Use Custom Data

There are three common approaches:

1. Prompting
2. Fine-Tuning
3. Retrieval-Augmented Generation (RAG)

---

# Prompting

The simplest way to provide custom information is through prompting.

This means supplying the information directly to the model as:

- System messages
- Few-shot examples
- Chat history
- User prompts
- Or any other appropriate prompt format

Advantages:

- Quick to implement
- Simple
- Larger context windows allow more information to be passed

Disadvantages:

- Long prompts consume many tokens
- Higher API cost
- Slower responses
- Performance decreases when prompts become excessively long

Use this technique carefully.

---

# Fine-Tuning

Fine-tuning continues training an already pre-trained language model (such as GPT-4) using your own dataset.

Unlike prompting, fine-tuning actually changes the model's internal neural network weights.

Benefits:

- Already trained on massive datasets
- Generally produces higher-quality responses
- Requires less prompt context
- Uses shorter prompts
- Can reduce response time

Disadvantages:

- Time consuming
- Computationally expensive
- Requires a large dataset
- Requires machine learning and deep learning expertise

---

# Retrieval-Augmented Generation (RAG)

Our focus in this section is Retrieval-Augmented Generation (RAG).

Instead of sending all of your data to the model, RAG stores embedded documents inside a vector database and retrieves only the information relevant to the user's question.

---

# Introduction to RAG

Previously we discussed:

### Prompting

- Fast
- Easy
- Suitable only for smaller datasets

### Fine-Tuning

- Produces faster and often more accurate models
- Lower cost during inference
- Requires significant resources

---

# Retrieval-Augmented Generation (RAG)

A RAG application consists of three major parts:

- Indexing
- Retrieval
- Generation

---

# Indexing

Indexing prepares your data for efficient retrieval.

The indexing process includes:

1. Loading documents
2. Splitting documents
3. Embedding documents
4. Storing embeddings

---

## 1. Document Loading

Load custom data from many different formats into LangChain's standard Document format.

Supported sources include:

- PDFs
- HTML
- JSON
- Google Drive
- Dropbox
- Many other file formats

---

## LangChain Document Object

A Document stores:

- Text
- Metadata

Example metadata:

- Page number
- Document title
- Subtitle
- Other useful information

Converting every supported file into a common Document format allows all later stages of the RAG pipeline to work identically regardless of the original source.

---

## Why Documents Must Be Split

Initially, documents may be extremely large.

This creates two major problems:

### Context Window Limits

Large documents may exceed an LLM's maximum context window.

### Token Consumption

Even if the document fits, repeatedly sending large documents:

- Costs more
- Takes longer
- Decreases efficiency

Therefore documents must be split into smaller chunks.

---

## Better Chunks

Studies show that language models perform better when given smaller chunks focused on a single topic rather than one very large document containing multiple unrelated topics.

Good chunks should:

- Fit inside the model's context window
- Cover one topic
- Preserve semantic meaning

---

## Why Embeddings?

Suppose we want a chatbot to answer questions using these document chunks.

Searching every chunk one by one would be inefficient.

Instead we embed every chunk.

---

# Embeddings

Embedding is the process of converting text into a numerical vector representation that captures its semantic meaning.

Once represented as vectors we can perform semantic similarity searches efficiently.

---

## Example

Imagine document chunks containing:

- Star
- Sun
- Spaghetti
- Bolognese
- Ice Cream
- Sorbet

After embedding:

- Star and Sun appear close together.
- Spaghetti and Bolognese appear close together.
- Ice Cream and Sorbet appear close together.

Star and Sun will be far away from Spaghetti and Bolognese because their meanings differ.

---

## User Question Example

Suppose the user asks:

> Can you give me a quick dinner idea to prepare this evening?

The query is also embedded.

Its vector will lie closest to:

- Spaghetti
- Bolognese

Next closest:

- Ice Cream
- Sorbet

Farthest:

- Star
- Sun

Therefore the retriever selects the dinner-related chunks.

---

# Measuring Similarity

How do we determine whether vectors are close?

Common similarity measures include:

- Dot Product
- Cosine Similarity
- Euclidean Distance

OpenAI recommends **Cosine Similarity**.

---

# Cosine Similarity

Cosine similarity measures the angle between two vectors.

Small angle:

- High cosine value
- Similar meaning

Large angle:

- Low cosine value
- Different meaning

Examples:

- 0° → cosine = 1
- 30° → ~0.87
- 45° → ~0.70
- 60° → 0.50
- 90° → 0

As the angle increases, semantic similarity decreases.

---

# Embedding Dimensions

Real language cannot be represented accurately using only two dimensions.

Embedding models typically use thousands of dimensions.

OpenAI currently recommends:

- text-embedding-3-small
  - 1,536 dimensions

- text-embedding-3-large
  - 3,072 dimensions

---

# Storing Embeddings

After generating embeddings we need somewhere to store them.

Regular relational databases are not designed for similarity searches.

Example SQL query:

```sql
SELECT *
FROM student_info
WHERE user_id = 12;