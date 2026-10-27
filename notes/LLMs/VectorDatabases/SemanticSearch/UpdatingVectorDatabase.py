import os

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. Updating the Vector Database
# =====================================================

# In this lesson, we'll discuss how to update
# and improve our existing vector database.
#
# Regardless of the embedding algorithm used,
# we cannot expect outstanding search results
# when working with poor-quality or incomplete data.
#
# This relates to the well-known saying:
#
# "Garbage in, garbage out."
#
# With limited course information, conducting
# a comprehensive semantic search becomes challenging.


# =====================================================
# 2. Limitations of Our Current Course Data
# =====================================================

# In our case, 365 Data Science courses often cover
# various topics across individual course sections.
#
# However, these topics may not be reflected
# in the overall course title or description.
#
# A course covering numerous topics still needs
# a concise and descriptive title.
#
# Making titles excessively long would reduce
# readability and potentially create design issues
# when displaying them on different screen sizes.
#
# Therefore, course titles alone cannot provide
# all the information needed for accurate
# semantic search.


# =====================================================
# 3. Enriching Our Vector Database
# =====================================================

# Fortunately, we have access to additional data:
#
# - Course section names
# - Course section descriptions
#
# We can incorporate this information into
# our vector database to improve search capabilities.
#
# This process enriches our dataset by providing
# additional context for our embedding model.
#
# Including relevant information can help
# the model generate more meaningful embeddings
# and improve the relevance of search results.


# =====================================================
# 4. Approaches to Organizing Course Data
# =====================================================

# There are two main approaches we can use
# to incorporate section-level information
# into our vector database.


# -----------------------------------------------------
# Approach 1: Aggregate Data at the Course Level
# -----------------------------------------------------

# The first approach is to combine all
# information about a course into one
# large block of text.
#
# This includes:
#
# - Course information
# - Course descriptions
# - Section names
# - Section descriptions
#
# We can then generate an embedding for
# the combined text and upsert it into Pinecone.
#
# With this approach, each course is represented
# by a single vector containing information
# about all its sections.


# -----------------------------------------------------
# Approach 2: Separate Courses and Sections
# -----------------------------------------------------

# The second approach is to create a separate
# vector database entry for each course section.
#
# Each entry combines information about
# the course with information about
# one of its sections.
#
# For example:
#
# Course: Machine Learning in Python
# Section: Clustering
#
# This course-section combination becomes
# an individual entry in the vector database.
#
# Rather than representing an entire course
# with one embedding, we generate embeddings
# for individual course sections.
#
# This allows us to search for more specific
# topics covered within a course.


# =====================================================
# 5. Choosing an Approach
# =====================================================

# For now, we'll explore the approach where
# courses and their sections are combined
# into individual database entries.
#
# This differs from aggregating every section
# into a single course-level embedding.
#
# At the conclusion of the course, we'll
# discuss the preferred approach for
# this particular use case.
#
# However, the best approach depends on:
#
# - The structure of the dataset
# - The information being searched
# - The requirements of the application
#
# Different applications may benefit
# from different approaches.


# =====================================================
# 6. Next Steps
# =====================================================

# In the next lesson, we'll read the
# additional course section data.
#
# We'll then create a separate entity
# for each course section and upload
# those entries to Pinecone.
#
# We'll also include metadata to help
# us identify and navigate the
# retrieved information more efficiently.


# =====================================================
# 7. Upserting Data to Pinecone
# =====================================================

# We've already explored how to generate embeddings
# from our course-section dataset.
#
# Now, we'll use those embeddings to create
# a new vector store in Pinecone.
#
# To accomplish this, we'll use the same
# Pinecone connection setup from our previous lessons.


# =====================================================
# 8. Loading the Course-Section Dataset
# =====================================================

# Load the course-section descriptions CSV file.
#
# The dataset uses Windows-1252 encoding.
#
# Each row represents a course-section combination.

files = pd.read_csv(
    "course_section_descriptions.csv",
    encoding="cp1252"
)


# =====================================================
# 9. Creating Unique Course-Section IDs
# =====================================================

# Create a unique identifier for each course section.
#
# We'll combine:
#
# - course_id
# - section_id
#
# A hyphen separates the two identifiers.
#
# For example:
#
# course_id = 15
# section_id = 3
#
# unique_id = "15-3"
#
# Converting both columns to strings allows
# us to concatenate them using the + operator.
#
# These unique IDs will identify our vectors
# when we upload them to Pinecone.

files["unique_id"] = (
    files["course_id"].astype(str)
    + "-"
    + files["section_id"].astype(str)
)


# =====================================================
# 10. Creating Metadata
# =====================================================

# We'll create a metadata dictionary for every row.
#
# Metadata contains additional information
# associated with a vector.
#
# Our metadata will include:
#
# - course_name
# - section_name
# - section_description
#
# Including this information allows us to
# identify courses and sections when performing
# similarity searches in Pinecone.
#
# We'll use pandas apply() with a lambda function.
#
# axis=1 means the lambda function processes
# each DataFrame row individually.
#
# The resulting dictionaries will be stored
# in the metadata column.

files["metadata"] = files.apply(
    lambda row: {
        "course_name": row["course_name"],
        "section_name": row["section_name"],
        "section_description": row["section_description"]
    },
    axis=1
)


# =====================================================
# 11. Creating Course-Section Embeddings
# =====================================================

# Next, we'll define a function that combines
# the course and section information into
# a single text string.
#
# The combined text will contain:
#
# - Course name
# - Course technology
# - Course description
# - Section name
# - Section description
#
# We're generating embeddings at the section level.
#
# This means course-level information will be
# repeated across multiple section embeddings.
#
# However, each section will also contain
# its own name and description.
#
# We'll use an f-string to combine the text
# and model.encode() to generate the embedding.


def create_embeddings(row):

    combined_text = f"""{row['course_name']} {row['course_technology']}
{row['course_description']} {row['section_name']} {row['section_description']}
"""

    # Convert the combined text into a numerical vector.
    #
    # show_progress_bar=False prevents the model
    # from displaying a progress bar for each row.

    return model.encode(
        combined_text,
        show_progress_bar=False
    )


# =====================================================
# 12. Loading the Embedding Model
# =====================================================

# We'll use the same Sentence Transformers model
# from our previous lessons.
#
# Model: all-MiniLM-L6-v2
#
# This model converts text into 384-dimensional
# numerical embeddings.
#
# Our Pinecone index must use the same dimension
# as the embeddings generated by this model.

model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# 13. Generating Embeddings for Every Section
# =====================================================

# Apply create_embeddings() to every row.
#
# axis=1 means each row is passed to the function.
#
# The resulting embeddings will be stored
# in the embedding column.
#
# Each embedding is a NumPy array containing
# 384 numerical values.

files["embedding"] = files.apply(
    create_embeddings,
    axis=1
)


# =====================================================
# 14. Loading Environment Variables
# =====================================================

# Load environment variables from the .env file.
#
# find_dotenv() searches for the .env file.
#
# override=True means values in the .env file
# can replace existing environment variables
# with the same names.

load_dotenv(
    find_dotenv(),
    override=True
)


# =====================================================
# 15. Initializing the Pinecone Client
# =====================================================

# Initialize the Pinecone client using
# the PINECONE_API_KEY environment variable.
#
# The client allows us to create, delete,
# and interact with Pinecone indexes.

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)


# =====================================================
# 16. Configuring the Pinecone Index
# =====================================================

# We'll use the same index configuration
# as our previous lessons.
#
# Index name: my-index
# Dimensions: 384
# Similarity metric: cosine
#
# The dimension must match the output
# of our embedding model.
#
# Cosine similarity measures how closely
# two vectors align in a multidimensional space.

index_name = "my-index"
dimension = 384
metric = "cosine"


# =====================================================
# 17. Deleting the Existing Index
# =====================================================

# Before creating our new index, we'll check
# whether an index with the same name exists.
#
# If it exists, we'll delete it.
#
# This allows us to rebuild the vector database
# using our new course-section embeddings.
#
# IMPORTANT:
#
# Deleting an index permanently removes
# its previously stored vectors and metadata.
#
# We're intentionally replacing our previous
# course-level vector database with the
# new section-level database.

if index_name in [
    index.name for index in pc.list_indexes()
]:

    pc.delete_index(index_name)

    print(f"{index_name} successfully deleted")

else:

    print(f"{index_name} not in index list.")


# =====================================================
# 18. Creating the New Pinecone Index
# =====================================================

# Create a new Pinecone index using:
#
# - The configured index name
# - 384 dimensions
# - Cosine similarity
# - AWS cloud infrastructure
# - The us-east-1 region
#
# The ServerlessSpec object defines
# the cloud provider and deployment region.

pc.create_index(
    name=index_name,
    dimension=dimension,
    metric=metric,
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


# =====================================================
# 19. Connecting to the Pinecone Index
# =====================================================

# Create an index object that we'll use
# to interact with our new vector database.
#
# This object provides methods such as:
#
# - upsert()
# - query()
# - delete()
#
# We'll use upsert() to upload our embeddings.

index = pc.Index(index_name)


# =====================================================
# 20. Preparing Vectors for Upserting
# =====================================================

# We're now ready to prepare our data
# for uploading to Pinecone.
#
# We'll use the embedding column containing
# our vectorized course-section information.
#
# Each vector requires:
#
# - A unique ID
# - Numerical embedding values
# - Metadata
#
# We'll use a list comprehension to iterate
# through the rows of our DataFrame.
#
# The iterrows() method returns each row
# along with its DataFrame index.
#
# For each row, we'll create a tuple containing:
#
# 1. The course-section unique ID
# 2. The embedding values
# 3. The metadata dictionary
#
# The tolist() method converts each NumPy
# embedding array into a standard Python list.
#
# Pinecone accepts these values when
# performing an upsert operation.

vectors_to_upsert = [
    (
        row["unique_id"],
        row["embedding"].tolist(),
        row["metadata"]
    )
    for _, row in files.iterrows()
]


# =====================================================
# 21. Upserting Vectors Into Pinecone
# =====================================================

# Once our vectors are prepared,
# we can upload them to Pinecone.
#
# We'll use the upsert() method.
#
# Upsert means:
#
# - Insert a vector if its ID doesn't exist.
# - Update a vector if its ID already exists.
#
# Depending on the dataset size, network connection,
# and available computing resources,
# uploading the vectors may take some time.
#
# After the operation completes, we can
# visit the Pinecone dashboard to inspect
# the stored vectors and their metadata.

index.upsert(
    vectors=vectors_to_upsert
)

print("Data Successfully upserted to Pinecone index.")


# =====================================================
# 22. Reviewing the Updated Vector Database
# =====================================================

# Our course-section embeddings have now
# been uploaded to Pinecone.
#
# Each vector represents a specific
# course-section combination.
#
# When we perform a similarity search,
# Pinecone can return the vectors that
# are most similar to our query.
#
# For example, if we request the top 10
# most similar vectors, the results can
# include metadata showing:
#
# - The course name
# - The section name
# - The section description
#
# This makes it easier to understand
# which course sections were retrieved
# and evaluate their relevance.