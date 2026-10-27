from pinecone import Pinecone, ServerlessSpec

import os
import time

from dotenv import load_dotenv, find_dotenv


# =====================================================
# Overview
# =====================================================

# In this lesson, we'll learn how to populate a Pinecone
# vector database using the upsert() method.
#
# We'll cover:
#
# 1. Connecting to Pinecone.
# 2. Checking whether an index already exists.
# 3. Creating an index if necessary.
# 4. Waiting for the index to become ready.
# 5. Connecting to an existing index.
# 6. Upserting vectors into the database.
# 7. Understanding the importance of vector dimensions.


# =====================================================
# Load Environment Variables
# =====================================================

# find_dotenv() searches for the .env file.
#
# load_dotenv() loads the environment variables from that file.
#
# override=True allows values from the .env file to overwrite
# existing environment variables with the same names.

load_dotenv(
    find_dotenv(),
    override=True
)


# =====================================================
# Connect to Pinecone
# =====================================================

# Initialize the Pinecone client using the API key
# stored in the environment variables.

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)


# =====================================================
# Define Index Configuration
# =====================================================

# Define the configuration for our Pinecone index.
#
# index_name:
#     The name of the index.
#
# dimension:
#     The number of numerical values in each vector.
#
# metric:
#     The similarity metric used to compare vectors.
#
# Our index will have:
#
# - Name: my-index
# - Dimensions: 3
# - Similarity metric: cosine

index_name = "my-index"
dimension = 3
metric = "cosine"


# =====================================================
# Check Whether the Index Exists
# =====================================================

# Before creating an index, check whether it already exists.
#
# pc.list_indexes().names() retrieves the names of the
# existing indexes.
#
# If my-index doesn't exist, we'll create it.
#
# Unlike our previous examples, we're not deleting an
# existing index before creating a new one.
#
# This allows us to preserve any existing data.

if index_name not in pc.list_indexes().names():

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
# Wait for the Index to Become Ready
# =====================================================

# After creating an index, Pinecone may need some time
# before the index is ready to accept operations.
#
# describe_index() retrieves information about the index.
#
# The status["ready"] value indicates whether the
# index is ready.
#
# If the index isn't ready:
#
# - Wait for one second.
# - Check the index status again.
# - Repeat until the index becomes ready.

while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)


# =====================================================
# Understanding Upsert
# =====================================================

# Now it's time for the next important phase:
#
# Populating our vector database with data.
#
# This process is called upserting.
#
# The term "upsert" combines two operations:
#
# - Update
# - Insert
#
# Upserting means:
#
# 1. If a vector with the specified ID already exists,
#    its data will be updated.
#
# 2. If a vector with the specified ID doesn't exist,
#    a new vector will be inserted.
#
# We'll use the term "upsert" throughout the rest
# of the course when adding or updating vector data.


# =====================================================
# Connect to the Index
# =====================================================

# Before upserting vectors, we need to create an index
# object that identifies which Pinecone index we'll use.
#
# We can accomplish this using the Index() method.
#
# We only need to specify the name of the index.

index = pc.Index(
    name=index_name
)


# =====================================================
# Upserting Vector Data
# =====================================================

# One of the simplest ways to upsert vector data is
# to create a list of tuples.
#
# Each tuple contains:
#
# 1. A unique ID identifying the vector.
# 2. A list containing the vector's numerical values.
#
# Since our index has three dimensions, each vector
# must contain exactly three numerical values.
#
# For this example, we'll use animal names as IDs.
#
# The three dimensions will represent:
#
# - Number of legs
# - Number of wings
# - Number of tails
#
# Example:
#
# ("Dog", [4., 0., 1.])
#
# This represents:
#
# - ID: Dog
# - Legs: 4
# - Wings: 0
# - Tails: 1
#
# The following example inserts five animal vectors
# into our Pinecone index.
#
# NOTE:
# The upsert() call is intentionally commented out,
# preserving the original lesson.

# print(index.upsert(vectors=[
#     ("Dog", [4., 0., 1.]),
#     ("Cat", [4., 0., 1.]),
#     ("Chicken", [2., 2., 1.]),
#     ("Mantis", [6., 2., 3.]),
#     ("Elephant", [4., 0., 1.]),
# ]))


# =====================================================
# Example Output
# =====================================================

# UpsertResponse(upserted_count=5)

# The response indicates that five vectors were
# successfully upserted into the index.
#
# If a vector ID already existed, the corresponding
# vector would be updated rather than duplicated.


# =====================================================
# Limitations of Our Animal Example
# =====================================================

# Looking back at our animal vectors, we can see
# that we've omitted important information.
#
# For example, we haven't included characteristics
# such as:
#
# - Animal genus
# - Animal family
#
# Without these additional characteristics, the
# similarity metric cannot evaluate those types
# of similarities.
#
# For example:
#
# Dog:
#     [4., 0., 1.]
#
# Cat:
#     [4., 0., 1.]
#
# Elephant:
#     [4., 0., 1.]
#
# All three animals have identical vector values.
#
# Therefore, the similarity metric cannot distinguish
# between them based on these three features alone.
#
# We'll explore the importance of representing data
# with sufficient information in later lessons.


# =====================================================
# Understanding Higher-Dimensional Vectors
# =====================================================

# If we wanted to continue developing our animal
# example, we could create an index with more dimensions.
#
# This would allow us to represent additional
# characteristics of each animal.
#
# As the number of characteristics increases, the
# number of dimensions may also increase.
#
# Suddenly, vectors containing 700 or 1500 dimensions
# don't seem so unusual.
#
# Higher-dimensional vectors can represent more
# complex information about our data.
#
# The important concept is that vector dimensions
# determine how much numerical information can
# be represented within each vector.