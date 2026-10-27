import os
import time

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. Introduction to Pinecone Indexing
# =====================================================

# Now that we've prepared our Pinecone connection
# and formatted our course descriptions, we're ready
# to generate vector embeddings.
#
# Embeddings are numerical representations of text
# that can capture semantic relationships.
#
# They are useful for applications such as:
#
# - Semantic search
# - Recommendation systems
# - Document retrieval
# - Similarity matching
#
# In this lesson, we'll:
#
# 1. Load our course dataset.
# 2. Prepare the course descriptions.
# 3. Connect to our Pinecone index.
# 4. Load a pretrained embedding model.
# 5. Generate embeddings for each course.
# 6. Prepare vectors for Pinecone.
# 7. Upsert the vectors into our index.


# =====================================================
# 2. Loading the Course Dataset
# =====================================================

# Load our course dataset using pandas.
#
# The CSV file uses Windows-1252 encoding.

files = pd.read_csv(
    "course_descriptions.csv",
    encoding="cp1252"
)


# =====================================================
# 3. Creating Course Descriptions
# =====================================================

# Reuse the function from our previous lessons.
#
# This function combines:
#
# - course_name
# - course_slug
# - course_technology
# - course_topic
#
# The resulting text provides additional context
# for our embedding model.


def create_course_description(row):
    return f"""The course name is {row["course_name"]}, the slug is {row["course_slug"]},
the technology is {row["course_technology"]} and the course topic is {row["course_topic"]}
"""


# =====================================================
# 4. Preparing the DataFrame
# =====================================================

# Configure pandas to display up to 106 rows.

pd.set_option("display.max_rows", 106)

# Apply our function to every row.
#
# axis=1 means the function receives one row at a time.

files["new_course_description"] = files.apply(
    create_course_description,
    axis=1
)


# =====================================================
# 5. Loading Environment Variables
# =====================================================

# Load the Pinecone API key from our .env file.
#
# find_dotenv() searches for the environment file.
#
# override=True gives values from the .env file
# priority over existing environment variables.

load_dotenv(find_dotenv(), override=True)

api_key = os.environ.get("PINECONE_API_KEY")

if not api_key:
    raise ValueError(
        "PINECONE_API_KEY was not found in the environment."
    )


# =====================================================
# 6. Initializing the Pinecone Client
# =====================================================

# Initialize the Pinecone client using our API key.
#
# This client allows us to manage indexes
# and interact with stored vectors.

pc = Pinecone(
    api_key=api_key
)


# =====================================================
# 7. Configuring Our Pinecone Index
# =====================================================

# We'll use the same configuration as PineconeSetup.py.
#
# Index name: my-index
# Dimensions: 384
# Similarity metric: cosine
#
# The dimension must match the output of
# our selected embedding model.

index_name = "my-index"
dimension = 384
metric = "cosine"


# =====================================================
# 8. Checking Whether the Index Exists
# =====================================================

# Before connecting to our index, check whether
# it already exists.
#
# Unlike PineconeSetup.py, we will NOT delete
# and recreate an existing index.
#
# Deleting the index would permanently remove
# any vectors that we've already uploaded.
#
# If the index does not exist, we'll create it.

existing_indexes = [
    index.name for index in pc.list_indexes()
]

if index_name not in existing_indexes:

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print(f"{index_name} successfully created")

else:

    print(f"{index_name} already exists")


# =====================================================
# 9. Waiting for the Index to Become Ready
# =====================================================

# A newly created Pinecone index may require
# some time before it can accept vector operations.
#
# We'll check its status before proceeding.
#
# is_ready indicates whether the index
# is available for operations.

while not pc.describe_index(index_name).status["ready"]:

    print("Waiting for Pinecone index to become ready...")

    time.sleep(2)

print(f"{index_name} is ready")


# =====================================================
# 10. Connecting to the Pinecone Index
# =====================================================

# Retrieve an index object.
#
# We'll use this object to upload our embeddings.

index = pc.Index(index_name)


# =====================================================
# 11. Loading the Embedding Model
# =====================================================

# The Sentence Transformers library provides
# pretrained models for generating embeddings.
#
# We'll use:
#
# all-MiniLM-L6-v2
#
# This model generates 384-dimensional vectors.
#
# It provides a useful balance between:
#
# - Embedding quality
# - Model size
# - Inference speed
# - Computational efficiency
#
# The generated vectors will represent
# the semantic content of our course descriptions.

model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# 12. Creating Course Embeddings
# =====================================================

# We need a function that combines the course
# description fields and generates an embedding.
#
# Our dataset contains:
#
# - course_description
# - new_course_description
# - course_description_short
#
# Combining these fields provides additional
# information about each course.
#
# The embedding model will encode this combined
# text into a numerical vector.
#
# NOTE:
#
# More text does not automatically guarantee
# better embeddings.
#
# The relevance, quality, and length of the
# input text also affect embedding quality.
#
# Long inputs may be truncated by the model.


def create_embedding(row):

    # Combine the three description fields
    # into one string.

    combined_text = " ".join(
        [
            str(row[field])
            for field in [
                "course_description",
                "new_course_description",
                "course_description_short"
            ]
        ]
    )

    # Convert the combined text into a vector.
    #
    # show_progress_bar=False prevents the
    # model from displaying a progress bar
    # for each individual course.

    embedding = model.encode(
        combined_text,
        show_progress_bar=False
    )

    return embedding


# =====================================================
# 13. Generating Embeddings for Every Course
# =====================================================

# Apply our embedding function to each row.
#
# axis=1 means that pandas passes each row
# into create_embedding().
#
# The resulting embeddings will be stored
# in a new DataFrame column.

files["embedding"] = files.apply(
    create_embedding,
    axis=1
)

# Each value in files["embedding"] is a
# NumPy array containing 384 numbers.
#
# Example structure:
#
# [0.012, -0.034, 0.056, ..., 0.021]
#
# The values above are illustrative.


# =====================================================
# 14. Preparing Vectors for Pinecone
# =====================================================

# Pinecone requires each vector to have
# a unique identifier.
#
# We'll use the course slug as the vector ID.
#
# This is generally more reliable than using
# the course name because course names
# may not always be unique.
#
# Each vector will contain:
#
# - id: Unique identifier
# - values: Numerical embedding
# - metadata: Additional course information
#
# The metadata allows us to identify courses
# when retrieving search results.
#
# The tolist() method converts a NumPy array
# into a standard Python list.
#
# Pinecone accepts lists of numerical values
# for vector upserts.

vectors_to_upsert = []

for row_number, row in files.iterrows():

    vector_id = str(row["course_slug"])

    vector_values = row["embedding"].tolist()

    vector_metadata = {
        "course_name": str(row["course_name"]),
        "course_slug": str(row["course_slug"]),
        "course_technology": str(row["course_technology"]),
        "course_topic": str(row["course_topic"])
    }

    vectors_to_upsert.append(
        {
            "id": vector_id,
            "values": vector_values,
            "metadata": vector_metadata
        }
    )


# =====================================================
# 15. Upserting Vectors Into Pinecone
# =====================================================

# Upsert means:
#
# - Insert a vector if its ID does not exist.
# - Update the vector if its ID already exists.
#
# This allows us to rerun the indexing process
# without automatically creating duplicate
# records for the same vector ID.
#
# We'll upload our course embeddings
# to the Pinecone index.
#
# The default namespace is used because
# we haven't specified a custom namespace.

index.upsert(
    vectors=vectors_to_upsert
)

print("Data upserted to Pinecone index")


# =====================================================
# 16. Verifying the Index
# =====================================================

# We can inspect the index statistics
# to see information about stored vectors.
#
# Pinecone updates may not appear immediately
# because vector operations can be eventually
# consistent.

stats = index.describe_index_stats()

print(stats)


# =====================================================
# 17. Summary
# =====================================================

# In this lesson, we learned how to:
#
# 1. Load and preprocess our course dataset.
# 2. Connect to an existing Pinecone index.
# 3. Create an index if one does not exist.
# 4. Wait for the index to become ready.
# 5. Load the all-MiniLM-L6-v2 embedding model.
# 6. Combine multiple course description fields.
# 7. Generate 384-dimensional embeddings.
# 8. Convert NumPy arrays into Python lists.
# 9. Prepare vector IDs and metadata.
# 10. Upsert vectors into Pinecone.
# 11. Inspect the index statistics.
#
# Our course embeddings are now stored
# in Pinecone and can be used for
# semantic similarity searches.
#
# In the next lesson, we'll generate an
# embedding for a search query and use
# Pinecone to retrieve relevant courses.