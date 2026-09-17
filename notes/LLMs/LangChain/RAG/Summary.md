# Retrieval-Augmented Generation (RAG) — Summary and Quick Reference

## What This Section Covers

Retrieval-Augmented Generation (RAG) gives a language model access to external
knowledge without retraining it. Instead of placing an entire knowledge base in
every prompt, the application retrieves the chunks most relevant to a question
and supplies only those chunks as context for the model's answer.

RAG is especially useful for private, current, or domain-specific information
that was not included in the model's training data.

## The Three Stages of RAG

### 1. Indexing

Indexing prepares source material for efficient search:

```text
source files -> load -> clean -> split -> embed -> vector store
```

- **Load:** Convert PDFs, DOCX files, and other sources into LangChain
  `Document` objects containing `page_content` and `metadata`.
- **Clean:** Remove formatting noise only when it does not carry useful
  structure.
- **Split:** Break large documents into focused chunks that fit comfortably in
  a model's context window.
- **Embed:** Convert each chunk into a numerical vector representing its
  semantic meaning.
- **Store:** Save text, vectors, metadata, and IDs in a vector database such as
  Chroma.

### 2. Retrieval

Retrieval finds the stored chunks that best match the user's question:

```text
question -> embed with the same model -> vector search -> relevant Documents
```

The query and stored documents must use the same embedding model so their
vectors can be compared in the same vector space.

### 3. Generation

Generation combines the original question with the retrieved context and asks
the language model to produce a grounded response:

```text
question + retrieved context -> prompt -> chat model -> parsed answer
```

## File Guide

- `RAGFundamentals.md` — Explains LLM knowledge limitations; compares
  prompting, fine-tuning, and RAG; and introduces indexing, retrieval, and
  generation.
- `DocumentLoading.py` — Loads PDF and DOCX files, examines LangChain
  `Document` objects, and cleans unnecessary whitespace.
- `DocumentSplitting.py` — Compares character-based splitting, chunk overlap,
  sentence separators, and Markdown-header splitting while preserving useful
  metadata.
- `TextEmbedding.py` — Embeds document chunks and uses dot products to compare
  their semantic similarity.
- `VectorStores.py` — Creates and reloads a persistent Chroma database and
  demonstrates retrieving, adding, updating, and deleting documents.
- `SimilaritySearch.py` — Retrieves the top matching chunks and shows how
  near-duplicates can consume multiple result slots.
- `MMRSearch.py` — Uses Maximal Marginal Relevance to balance result relevance
  and diversity, and demonstrates metadata filtering.
- `VectorStoreRetriever.py` — Converts Chroma into an LCEL-compatible
  `VectorStoreRetriever` and configures its search strategy.
- `StuffingDocuments.py` — Builds a complete LCEL chain that places retrieved
  documents directly into a prompt.
- `GeneratingResponse.py` — Completes the generation pipeline with
  `ChatOpenAI` and `StrOutputParser` and discusses alternatives to stuffing.

The PDF and DOCX files provide the example course content, while
`intro-to-ds-lectures/` contains the persisted Chroma vector store produced by
the indexing examples.

## Chunking and Metadata

Chunk size is a tradeoff:

- Smaller chunks are more focused and may improve retrieval precision, but can
  lose surrounding context and create more vectors.
- Larger chunks preserve more context, but may mix topics, increase prompt
  size, and make matches less precise.
- Chunk overlap preserves information across boundaries, but increases storage
  and can create repetitive search results.

Structure-aware splitting is often preferable when a document has meaningful
headings. In these lessons, Markdown headers create lecture-level documents,
then character splitting creates smaller chunks. Course and lecture titles are
kept as metadata so results can be filtered and sources can be identified.

## Similarity Search vs. MMR

| Method | Main goal | Main tradeoff |
| --- | --- | --- |
| Similarity search | Return the chunks closest to the query | Results may be repetitive |
| MMR search | Balance relevance with diversity | More diversity can reduce relevance |

For MMR, `lambda_mult` controls the balance:

- Values closer to `1` favor relevance.
- Lower values favor diversity.
- A value of `1` effectively removes the diversity component.

The `k` setting controls how many chunks are returned. Metadata filters can
restrict results to a course, lecture, source, or other known subset.

## Complete LangChain Workflow

The final chain follows this shape:

```text
                           -> retriever -> context -
user question -> parallel                         -> prompt
                           -> passthrough -> question -

prompt -> ChatOpenAI -> AIMessage -> StrOutputParser -> string answer
```

A compact LCEL version is:

```python
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | chat
    | StrOutputParser()
)
```

The prompt should tell the model to use the retrieved context, define how to
handle missing information, and request source information when citations or
traceability matter.

## Supplying Documents to the Model

The examples use **stuffing**, which inserts all retrieved chunks into one
prompt. It is simple and often effective when the selected context fits within
the model's context window. Its limits are higher token usage, context-window
overflow, and the possibility that information in the middle receives less
attention.

For larger result sets, alternatives include processing documents
individually and refining an answer, or summarizing/interpreting chunks before
combining their results.

## Key Takeaways

- RAG adds external knowledge at request time; it does not change model
  weights.
- Retrieval quality depends heavily on loading, cleaning, chunking, metadata,
  the embedding model, and search configuration.
- Use the same embedding model for indexed chunks and incoming queries.
- Similarity search maximizes relevance; MMR can reduce redundant results.
- Preserve metadata because it supports filtering, attribution, debugging, and
  source display.
- Retrieve only enough context to answer the question well; excess context
  increases cost and can reduce answer quality.
- Grounding a prompt in retrieved text reduces unsupported answers, but the
  application still needs evaluation and clear behavior for missing evidence.

## Terms to Remember

- **Document:** LangChain's container for text (`page_content`) and associated
  `metadata`.
- **Chunk:** A smaller section of a source document used as the unit of
  embedding and retrieval.
- **Embedding:** A fixed-length numerical representation of semantic meaning.
- **Vector store:** A database that stores embeddings and supports nearest-
  neighbor search.
- **Similarity search:** Retrieval based on closeness between query and
  document vectors.
- **MMR:** Maximal Marginal Relevance, a method that balances relevance and
  diversity.
- **Retriever:** A Runnable that accepts a query and returns relevant
  `Document` objects.
- **Stuffing:** Placing all retrieved documents into a single model prompt.
- **Grounding:** Constraining or guiding an answer with retrieved evidence.
