from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os


# =====================================================
# Introduction
# =====================================================

# In previous lessons, we learned how to manage Pinecone indexes
# and upsert vector data.
#
# In this lesson, we'll learn how to:
#
# 1. Load a large text dataset from Hugging Face.
# 2. Convert text into numerical embeddings.
# 3. Create a Pinecone index with the correct dimensions.
# 4. Prepare vectors and metadata for uploading.
# 5. Upsert vectors into Pinecone in batches.
#
# We'll use the FineWeb dataset from Hugging Face.
#
# FineWeb contains over 15 trillion tokens of English web data
# sourced from Common Crawl.
#
# Each dataset record contains information such as:
#
# - Text
# - ID
# - URL
# - Date
# - Language
# - Language score
# - Token count
#
# The dataset contains documents of different lengths, making
# it useful for demonstrating how vector databases handle
# large collections of text.


# =====================================================
# 1. Loading the FineWeb Dataset
# =====================================================

# Use load_dataset() to load the FineWeb dataset.
#
# We're using:
#
# Dataset: HuggingFaceFW/fineweb
# Configuration: sample-10BT
# Split: train
#
# Setting streaming=True returns an IterableDataset instead
# of downloading the entire dataset into memory.

fw = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True
)

# View the dataset structure.
# print(fw)


# =====================================================
# 2. Understanding IterableDataset
# =====================================================

# An IterableDataset allows us to access records sequentially
# without loading the entire dataset into memory.
#
# This is particularly useful when working with datasets
# that are too large to fit into our computer's RAM.
#
# We can iterate through the dataset using a for loop.
#
# Each iteration retrieves another record.
#
# Example output:
#
# IterableDataset({
#     features: [
#         'text',
#         'id',
#         'dump',
#         'url',
#         'date',
#         'file_path',
#         'language',
#         'language_score',
#         'token_count'
#     ],
#     num_shards: 15
# })
#
# The number of shards may vary depending on the dataset version.

# View the dataset features.
# print(fw.features)

# Example output:
#
# {
#     'text': Value('string'),
#     'id': Value('string'),
#     'dump': Value('string'),
#     'url': Value('string'),
#     'date': Value('string'),
#     'file_path': Value('string'),
#     'language': Value('string'),
#     'language_score': Value('float64'),
#     'token_count': Value('int64')
# }


# =====================================================
# 3. Understanding Text Embeddings
# =====================================================

# Before uploading our dataset to Pinecone, we must convert
# the text into numerical vectors.
#
# This process is called embedding.
#
# An embedding model transforms text into a vector of numbers
# that represents information about the text's meaning.
#
# Text with similar meanings will generally produce vectors
# that are close together in the embedding space.
#
# This allows vector databases to perform semantic searches
# instead of relying exclusively on exact keyword matches.
#
# We'll use the Sentence Transformers library.
#
# Model: all-MiniLM-L6-v2
#
# This model is relatively fast and produces embeddings
# containing 384 dimensions.


# =====================================================
# 4. Loading the Embedding Model
# =====================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# The model converts text into numerical embeddings.
#
# Example:
#
# embedding = model.encode("This is an example sentence.")
#
# The result is a NumPy ndarray containing 384 values.
#
# We can convert the ndarray into a Python list using:
#
# embedding.tolist()
#
# Pinecone accepts vectors containing numerical values,
# so converting the embeddings into lists is convenient
# when preparing the upsert request.


# =====================================================
# 5. Connecting to Pinecone
# =====================================================

# Load environment variables from the .env file.
#
# override=True means values in the .env file can replace
# existing environment variables with the same names.

load_dotenv(find_dotenv(), override=True)

# Initialize the Pinecone client using our API key.

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# View the indexes currently available in Pinecone.
# print(pc.list_indexes().to_dict())


# =====================================================
# 6. Reviewing Existing Pinecone Indexes
# =====================================================

# In previous lessons, we created three indexes:
#
# 1. my-index
#    - Dimensions: 3
#    - Metric: cosine
#
# 2. my-newnew-index
#    - Dimensions: 1536
#    - Metric: euclidean
#
# 3. testingcreatingnewindex
#    - Dimensions: 1536
#    - Metric: euclidean
#
# These indexes demonstrate that Pinecone supports
# different vector dimensions and similarity metrics.
#
# However, the dimensions of an index must match the
# dimensions produced by the embedding model.
#
# For this lesson, we'll create a new index named "text".


# =====================================================
# 7. Understanding Embedding Dimensions
# =====================================================

# Unlike our earlier examples, we cannot choose an arbitrary
# number of dimensions for this index.
#
# The dimensions must match the output of our embedding model.
#
# The all-MiniLM-L6-v2 model generates 384-dimensional vectors.
#
# Therefore, our Pinecone index must also have 384 dimensions.
#
# We could hardcode the value:
#
# dimension = 384
#
# However, using get_sentence_embedding_dimension() is better
# because it retrieves the dimension directly from the model.
#
# This makes our code easier to reuse with other models.

dimension = model.get_sentence_embedding_dimension()

# print(dimension)

# Expected output:
#
# 384


# =====================================================
# 8. Creating the Pinecone Index
# =====================================================

# Create an index named "text".
#
# Parameters:
#
# name:
#     The name of our Pinecone index.
#
# dimension:
#     The number of dimensions in each embedding.
#
# metric:
#     The similarity metric used to compare vectors.
#
# spec:
#     The cloud provider and region where the index
#     will be hosted.
#
# We'll use cosine similarity because it is commonly
# used when comparing text embeddings.
#
# Check whether the index already exists before creating it.
#
# This prevents an error when running the script again.
#
# Unlike deleting and recreating the index, this approach
# preserves any vectors already stored in it.

index_name = "text"

if index_name not in [index.name for index in pc.list_indexes()]:

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print(f"{index_name} index created successfully.")

else:
    print(f"{index_name} index already exists.")


# =====================================================
# 9. Connecting to the Text Index
# =====================================================

# Connect to our newly created Pinecone index.
#
# We'll use this index to upload our text embeddings.

index = pc.Index(name=index_name)

# Note:
#
# Pinecone indexes may take some time to become ready
# after creation.
#
# If an index was just created, we should ensure that
# its status is ready before attempting to upsert data.

import time

while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)


# =====================================================
# 10. Preparing Data for Upserting
# =====================================================

# Now we're ready to convert our dataset into embeddings
# and upload them to Pinecone.
#
# First, we'll create a list to store the vectors.
#
# Each vector will contain:
#
# 1. A unique ID
# 2. An embedding
# 3. Metadata
#
# Pinecone accepts vectors in the following format:
#
# (
#     "unique-id",
#     [0.12, 0.45, 0.78, ...],
#     {"language": "en"}
# )
#
# The embedding contains the numerical representation
# of the original text.
#
# Metadata provides additional information that can
# be used when filtering search results.


# =====================================================
# 11. Limiting the Dataset Size
# =====================================================

# FineWeb is an extremely large dataset.
#
# Processing the entire dataset would require significant
# time, computing resources, and Pinecone storage.
#
# To make this lesson manageable, we'll process only
# 10,000 records.
#
# The subset_size variable controls the number of
# records we want to process.
#
# IMPORTANT:
#
# This selects the first 10,000 records encountered
# by the dataset iterator.
#
# It does not randomly sample 10,000 records.

subset_size = 10000


# =====================================================
# 12. Generating Embeddings
# =====================================================

# Create an empty list to store our vectors.

vectors_to_upsert = []

# Iterate through the FineWeb dataset.
#
# enumerate() provides:
#
# i:
#     The current iteration number.
#
# item:
#     The current dataset record.
#
# We'll stop once we've processed 10,000 records.

for i, item in enumerate(fw):

    # Stop processing once we reach the subset size.

    if i >= subset_size:
        break

    # Retrieve the text from the current record.

    text = item["text"]

    # Retrieve the unique ID.
    #
    # Convert it into a string because Pinecone
    # requires vector IDs to be strings.

    unique_id = str(item["id"])

    # Retrieve the language associated with the text.

    language = item["language"]

    # Convert the text into a numerical embedding.
    #
    # model.encode() returns a NumPy ndarray.
    #
    # show_progress_bar=False disables the progress bar.
    #
    # .tolist() converts the ndarray into a Python list.

    embedding = model.encode(
        text,
        show_progress_bar=False
    ).tolist()

    # Prepare metadata.
    #
    # We're storing the language associated with
    # each text record.

    metadata = {
        "language": language
    }

    # Append the vector to our list.
    #
    # Each entry contains:
    #
    # (unique_id, embedding, metadata)

    vectors_to_upsert.append(
        (
            unique_id,
            embedding,
            metadata
        )
    )


# =====================================================
# 13. Understanding Batch Upserts
# =====================================================

# Now that our embeddings are prepared, we can upload
# them to Pinecone.
#
# Instead of uploading each vector individually,
# we'll upload multiple vectors in batches.
#
# Batching reduces the number of API requests and
# can improve upload efficiency.
#
# For example, with 10,000 vectors and a batch size
# of 1,000:
#
# Batch 1:    Vectors 0-999
# Batch 2:    Vectors 1000-1999
# Batch 3:    Vectors 2000-2999
# ...
# Batch 10:   Vectors 9000-9999
#
# However, Pinecone also enforces a maximum request size.
#
# A batch containing 1,000 vectors may exceed that limit
# depending on the dimensions and metadata.
#
# We'll use a smaller batch size of 100 for this example.


# =====================================================
# 14. Upserting Vectors into Pinecone
# =====================================================

# Define the number of vectors to upload per batch.

batch_size = 100

# Iterate through our vector list in batches.
#
# range() starts at 0 and increases by batch_size.
#
# For each iteration:
#
# 1. Select a slice of vectors.
# 2. Store that slice in the batch variable.
# 3. Upload the batch to Pinecone.
#
# The slicing syntax:
#
# vectors_to_upsert[i:i + batch_size]
#
# selects up to batch_size vectors starting at position i.

for i in range(0, len(vectors_to_upsert), batch_size):

    batch = vectors_to_upsert[i:i + batch_size]

    # Upload the current batch to Pinecone.

    index.upsert(
        vectors=batch
    )

print("Subset of data upserted to Pinecone index.")


# =====================================================
# 15. Potential Drawbacks
# =====================================================

# Although this approach works for demonstrating the
# embedding and upsert process, it has some drawbacks.
#
# 1. Processing Time
#
#    Generating embeddings for 10,000 text records
#    can take a considerable amount of time.
#
#    The execution time depends on the computer's
#    hardware and the length of each document.
#
# 2. Memory Usage
#
#    Although the dataset is streamed, we still store
#    all 10,000 embeddings in vectors_to_upsert.
#
#    This consumes memory.
#
#    A more scalable approach would generate and
#    upload embeddings incrementally in batches.
#
# 3. Large Documents
#
#    The embedding model has an input token limit.
#
#    Text exceeding this limit may be truncated,
#    meaning the embedding may not represent the
#    entire original document.
#
#    Splitting long documents into smaller chunks
#    can help address this limitation.
#
# 4. Processing Speed
#
#    We're generating embeddings one record at a time.
#
#    Encoding multiple texts together in batches
#    can improve performance.
#
# 5. Pinecone Storage and API Usage
#
#    Uploading large datasets consumes Pinecone
#    storage and API resources.
#
#    We should consider usage limits and costs
#    before processing larger datasets.


# =====================================================
# 16. Summary
# =====================================================

# In this lesson, we learned how to:
#
# 1. Load the FineWeb dataset using Hugging Face.
#
# 2. Stream a large dataset using IterableDataset.
#
# 3. Load the all-MiniLM-L6-v2 embedding model.
#
# 4. Convert text into 384-dimensional embeddings.
#
# 5. Create a Pinecone index with matching dimensions.
#
# 6. Prepare vector IDs, embeddings, and metadata.
#
# 7. Limit processing to 10,000 records.
#
# 8. Upload embeddings to Pinecone in batches.
#
# 9. Understand the limitations of processing large datasets.
#
# In the next lesson, we'll use a smaller dataset
# for a more convenient case study.