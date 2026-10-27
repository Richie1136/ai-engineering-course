# Semantic Search — Summary

This folder explores semantic search through a course-discovery case study: helping students find relevant courses and sections even when their queries do not match the catalog's wording. The lessons use pandas, Sentence Transformers, and Pinecone to prepare data, generate embeddings, and retrieve similar content.

## 1. Core Concepts

- **Semantic search** retrieves content based on meaning and context. For example, a query about unsupervised learning can match a section about K-means clustering.
- **Embeddings** represent text as numerical vectors. An embedding model places related content near one another in its vector space.
- **Vector databases** store embeddings and efficiently retrieve similar vectors. Traditional databases remain useful for structured records and can work alongside them.
- **Semantic search retrieves existing information.** Generative AI produces new text and can use retrieved information as context, but can still generate unsupported claims.

See [Semantic Search Fundamentals](SemanticSearchFundamentals.md).

## 2. The Search Pipeline

**Course data → Text preparation → Embeddings → Pinecone index → Query embedding → Ranked results**

1. Load course data from CSV files using pandas. The examples use Windows-1252 (`cp1252`) encoding.
2. Combine relevant fields, such as course name, technology, topic, and descriptions, into text suitable for embedding.
3. Generate embeddings with a Sentence Transformers model.
4. Store each embedding with a unique ID and useful metadata in Pinecone.
5. Encode the user's query with the same model used for the stored embeddings.
6. Query the index, retrieve the closest matches, and display their metadata.
7. Review relevance and refine the data representation, model, or search settings.

## 3. Preparing and Indexing Data

[DataPreprocessing.py](DataPreprocessing.py) demonstrates combining fields with f-strings and applying functions to DataFrame rows using `apply(axis=1)`.

[PineconeSetup.py](PineconeSetup.py) introduces environment variables, the `PINECONE_API_KEY`, and index configuration. The initial example uses a 384-dimensional index with cosine similarity, hosted on AWS in `us-east-1`.

[PineconeIndexing.py](PineconeIndexing.py) generates embeddings with `all-MiniLM-L6-v2`, prepares vector IDs and metadata, uploads vectors with `upsert()`, and inspects index statistics. An upsert inserts a new ID or replaces the vector associated with an existing ID.

The index dimension must match the embedding output. Matching dimensions alone is insufficient: stored vectors and query vectors must also use a compatible embedding model and representation.

## 4. Searching and Evaluating Results

[SimilaritySearch.py](SimilaritySearch.py) encodes a query, calls `index.query()`, and displays matching course or section information.

- `top_k` controls how many nearest matches are requested.
- `include_metadata=True` returns information that makes the matches understandable.
- A score threshold can remove weaker matches from the displayed results.
- Similarity scores measure vector similarity; they are not probabilities that a result is relevant.
- Inspect the actual course and section descriptions when judging result quality. Thresholds need to be evaluated for the chosen model and dataset.

## 5. Improving Search Granularity

A single course embedding can obscure topics covered only in individual sections. [UpdatingVectorDatabase.py](UpdatingVectorDatabase.py) extends the dataset to represent each course-section combination separately.

Each section entry contains:

- A unique ID formed from `course_id` and `section_id`.
- An embedding incorporating course context and section-specific text.
- Metadata containing the course name, section name, and section description.

This allows results to point directly to relevant sections. An alternative is to aggregate all section information into one course representation; the useful level of detail depends on what users need to find.

## 6. Embedding Models and Weighted Features

[EmbeddingAlgorithms.py](EmbeddingAlgorithms.py) discusses text-vectorization approaches and experiments with alternative embeddings. Bag of Words and TF-IDF capture word occurrence or importance, while learned embeddings support comparisons based on semantic relationships.

[WeightedFeatures.py](WeightedFeatures.py) embeds fields separately and combines them using a weighted average:

| Field | Example weight |
|---|---:|
| Course name | 5 |
| Section name | 3 |
| Section description | 2 |
| Course technology | 1 |
| Course description | 1 |

The example uses `multi-qa-distilbert-cos-v1` and a separate index named `weighted-features-768`. Weighting gives selected fields more influence over the combined representation. These weights are experimental choices, so their effect should be assessed with representative queries.

Changing the embedding model requires regenerating stored embeddings and using the corresponding model for queries. Higher dimensionality or more input text does not automatically produce better results.

## 7. Applications Beyond Course Search

[Semantic Search Applications](SemanticSearchApplications.md) describes other uses of vector similarity:

- **Recommendations:** compare item characteristics or user preferences to retrieve related products and content.
- **Image search:** compare image embeddings to find visually or semantically similar images and products.
- **Biomedical research:** compare representations of gene expression, molecular structures, or biological activity to identify candidates for further investigation.

Across these applications, useful results depend on how well the representation captures the relationships being searched.

## 8. Working With the Examples

The scripts reference `course_descriptions.csv` or `course_section_descriptions.csv` and use a Pinecone API key loaded from the environment. They are instructional examples with overlapping setup steps, rather than a single sequential application.

`PineconeSetup.py` and `UpdatingVectorDatabase.py` include deletion and recreation of an existing index. Review those operations before running the examples against an index containing data you want to retain.
