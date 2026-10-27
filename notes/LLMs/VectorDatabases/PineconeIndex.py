from pinecone import Pinecone, ServerlessSpec

import os
from dotenv import load_dotenv, find_dotenv


# =====================================================
# Overview
# =====================================================

# In the previous lesson, we learned how to create and delete indexes
# using the Pinecone platform.
#
# Since we'll be doing most of our project work in Python, it's useful
# to understand how to manage Pinecone indexes directly through code.
#
# In this lesson, we'll learn how to:
#
# 1. Connect to Pinecone using Python.
# 2. Retrieve information about existing indexes.
# 3. Check whether an index already exists.
# 4. Delete an existing index.
# 5. Create new indexes with different configurations.


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

# Retrieve the Pinecone API key from the environment variables.
#
# Initialize the Pinecone client using that API key.

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)


# =====================================================
# List Existing Indexes
# =====================================================

# Call list_indexes() to retrieve the indexes currently managed
# by Pinecone.
#
# to_dict() converts the response into a Python dictionary.

# print(pc.list_indexes().to_dict())


# =====================================================
# Define the Index Configuration
# =====================================================

# Before creating an index, define the following variables:
#
# - index_name: The name of the index.
# - dimension: The number of dimensions in each vector.
# - metric: The similarity metric used to compare vectors.
#
# In this example, we'll create an index with:
#
# Name: my-index
# Dimensions: 3
# Similarity metric: cosine

index_name = "my-index"
dimension = 3
metric = "cosine"


# =====================================================
# Check Whether the Index Already Exists
# =====================================================

# As a practice exercise, we'll check whether an index with the
# specified name already exists before creating it.
#
# If the index exists, we'll delete it first.
#
# We can accomplish this using an if statement and a list comprehension.
#
# The list_indexes() method retrieves the indexes currently managed
# by Pinecone.
#
# Each item represents an index object.
#
# The list comprehension iterates over those indexes and retrieves
# their names.
#
# Example:
#
# [index.name for index in pc.list_indexes()]
#
# This produces a list containing the names of the existing indexes.
#
# We then check whether index_name appears in that list.
#
# If it does:
#     Delete the existing index.
#
# Otherwise:
#     Print a message indicating that the index doesn't exist.
#
# WARNING:
# Deleting an index permanently removes its stored vectors and data.

if index_name in [index.name for index in pc.list_indexes()]:

    pc.delete_index(index_name)

    print(
        f"{index_name} successfully deleted"
    )

else:

    print(
        f"{index_name} not in index list."
    )


# =====================================================
# Create the First Pinecone Index
# =====================================================

# Use create_index() to create a new index.
#
# The method accepts several parameters:
#
# name:
#     The name of the index.
#
# dimension:
#     The number of dimensions for each vector.
#
# metric:
#     The similarity metric used to compare vectors.
#
# spec:
#     The deployment configuration for the index.
#
# ServerlessSpec specifies the cloud provider and region.
#
# In this example:
#
# - Cloud provider: AWS
# - Region: us-east-1
#
# Using variables makes it easier to change the index configuration
# without modifying the create_index() call itself.

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
# Create the Second Pinecone Index
# =====================================================

# Create an additional index with a different configuration.
#
# This index uses:
#
# - Name: testingcreatingnewindex
# - Dimensions: 1536
# - Similarity metric: euclidean
# - Cloud provider: AWS
# - Region: us-east-1
#
# Unlike the first index, this example passes the configuration
# values directly into create_index().
#
# NOTE:
# This example does not check whether the second index already exists.
# If it already exists, Pinecone may raise an error when creating it.

pc.create_index(
    name="testingcreatingnewindex",
    dimension=1536,
    metric="euclidean",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


# =====================================================
# Verify the Created Indexes
# =====================================================

# Call list_indexes() again to retrieve information about
# the indexes after creating them.

# print(pc.list_indexes().to_dict())


# =====================================================
# Example Output: Newly Created Index
# =====================================================

# The following output shows information about the newly created
# testingcreatingnewindex index.
#
# Important properties:
#
# - Name: testingcreatingnewindex
# - Status: Ready
# - Dimension: 1536
# - Metric: euclidean
# - Cloud provider: AWS
# - Region: us-east-1

# {'name': 'testingcreatingnewindex', 'status': {'ready': True, 'state': 'Ready'}, 'schema': {'fields': {'_sparse_values':
# {'type': 'sparse_vector', 'description': None}, '_values': {'type': 'dense_vector', 'dimension': 1536, 'metric': 'euclidean',
# 'description': None}}}, 'deployment': {'deployment_type': 'managed', 'cloud': 'aws', 'region': 'us-east-1', 'environment': 'aped-4627-b74a'},
# 'deletion_protection': 'disabled', 'host': 'https://testingcreatingnewindex-6rd4bb8.svc.aped-4627-b74a.pinecone.io', 'read_capacity':
# {'mode': 'OnDemand', 'status': {'state': 'Ready', 'current_shards': None, 'current_replicas': None, 'error_message': None}}, 'tags': None,
# 'private_host': None, 'source_collection': None, 'source_backup_id': None, 'cmek_id': None, 'dimension': 1536, 'metric': 'euclidean',
# 'vector_type': 'dense', 'spec': {'serverless': {'cloud': 'aws', 'region': 'us-east-1', 'read_capacity': {'mode': 'OnDemand', 'status':
# {'state': 'Ready', 'current_shards': None, 'current_replicas': None, 'error_message': None}}, 'source_collection': None, 'schema': {'fields':
# {'_sparse_values': {'type': 'sparse_vector', 'description': None}, '_values': {'type': 'dense_vector', 'dimension': 1536, 'metric': 'euclidean',
# 'description': None}}}}, 'pod': None, 'byoc': None}, 'embed': None}]}


# =====================================================
# Summary
# =====================================================

# In this lesson, we learned how to manage Pinecone indexes using Python.
#
# We:
#
# 1. Loaded the Pinecone API key from the .env file.
# 2. Connected to Pinecone using the Pinecone client.
# 3. Retrieved existing indexes using list_indexes().
# 4. Used a list comprehension to check whether an index exists.
# 5. Deleted an existing index using delete_index().
# 6. Created two indexes using create_index().
#
# The first index uses:
#
# - 3 dimensions
# - Cosine similarity
#
# The second index uses:
#
# - 1536 dimensions
# - Euclidean distance
#
# These examples demonstrate how to manage index configurations
# directly through Python rather than the Pinecone interface.