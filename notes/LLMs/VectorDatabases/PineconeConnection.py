import pinecone
from pinecone import Pinecone, ServerlessSpec

import os
from dotenv import load_dotenv, find_dotenv


# =====================================================
# Overview
# =====================================================

# In this lesson, we'll learn how to connect to Pinecone using Python.
#
# We'll also explore how to:
#
# 1. Load the Pinecone API key from an environment file.
# 2. Establish a connection to Pinecone.
# 3. Retrieve information about existing indexes.
# 4. Delete an existing index.
# 5. Create a new index using Python.


# =====================================================
# Load Environment Variables
# =====================================================

# find_dotenv() searches for the .env file.
#
# load_dotenv() loads the environment variables from that file.
#
# Setting override=True means that values from the .env file will
# overwrite environment variables with the same names that are
# already present in the current process.

load_dotenv(
    find_dotenv(),
    override=True
)


# =====================================================
# Connect to Pinecone
# =====================================================

# Retrieve the Pinecone API key from the environment variables.
#
# Initialize the Pinecone client using the API key.

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)


# =====================================================
# List Existing Pinecone Indexes
# =====================================================

# Call list_indexes() to verify that the connection was successful.
#
# The method retrieves information about the indexes associated
# with our Pinecone account.
#
# to_dict() converts the response into a Python dictionary.
#
# If no indexes exist, the response will contain an empty data list.
#
# Otherwise, the response includes information such as:
#
# - Index names
# - Dimensions
# - Similarity metrics
# - Deployment information
# - Index status

# print(pc.list_indexes().to_dict())


# =====================================================
# Example Output: Existing Index
# =====================================================

# In this example, the account contains an index named my-index.
#
# The index has:
#
# - Name: my-index
# - Dimension: 3
# - Similarity metric: cosine
# - Cloud provider: AWS
# - Region: us-east-1
# - Status: Ready

# {'data': [{'name': 'my-index', 'status': {'ready': True, 'state': 'Ready'},
# 'schema': {'fields': {'_sparse_values': {'type': 'sparse_vector', 'description': None},
# '_values': {'type': 'dense_vector', 'dimension': 3, 'metric': 'cosine', 'description': None}}},
# 'deployment': {'deployment_type': 'managed', 'cloud': 'aws', 'region': 'us-east-1', 'environment':
# 'aped-4627-b74a'}, 'deletion_protection': 'disabled',
# 'host': 'https://my-index-6rd4bb8.svc.aped-4627-b74a.pinecone.io', 'read_capacity':
# {'mode': 'OnDemand', 'status': {'state': 'Ready', 'current_shards': None, 'current_replicas': None,
# 'error_message': None}}, 'tags': None, 'private_host': None, 'source_collection': None,
# 'source_backup_id': None, 'cmek_id': None, 'dimension': 3, 'metric': 'cosine',
# 'vector_type': 'dense', 'spec': {'serverless': {'cloud': 'aws', 'region': 'us-east-1',
# 'read_capacity': {'mode': 'OnDemand', 'status': {'state': 'Ready', 'current_shards': None,
# 'current_replicas': None, 'error_message': None}}, 'source_collection': None, 'schema': {'fields':
# {'_sparse_values': {'type': 'sparse_vector', 'description': None}, '_values':
# {'type': 'dense_vector', 'dimension': 3, 'metric': 'cosine', 'description': None}}}},
# 'pod': None, 'byoc': None}, 'embed': None}]}


# =====================================================
# Example Output: Another Index
# =====================================================

# print(pc.list_indexes().to_dict())

# After creating another index named new-index through Pinecone,
# we can retrieve its information using list_indexes().
#
# This index has:
#
# - Name: new-index
# - Dimension: 1536
# - Similarity metric: euclidean
# - Cloud provider: AWS
# - Region: us-east-1
# - Status: Ready

# {'name': 'new-index', 'status': {'ready': True, 'state': 'Ready'}, 'schema': {'fields':
# {'_values': {'type': 'dense_vector', 'dimension': 1536, 'metric': 'euclidean', 'description': None},
# '_sparse_values': {'type': 'sparse_vector', 'description': None}}}, 'deployment':
# {'deployment_type': 'managed', 'cloud': 'aws', 'region': 'us-east-1', 'environment': 'aped-4627-b74a'},
# 'deletion_protection': 'disabled', 'host': 'https://new-index-6rd4bb8.svc.aped-4627-b74a.pinecone.io',
# 'read_capacity': {'mode': 'OnDemand', 'status': {'state': 'Ready', 'current_shards': None,
# 'current_replicas': None, 'error_message': None}}, 'tags': None, 'private_host': None,
# 'source_collection': None, 'source_backup_id': None, 'cmek_id': None, 'dimension': 1536,
# 'metric': 'euclidean', 'vector_type': 'dense', 'spec': {'serverless': {'cloud': 'aws', 'region':
# 'us-east-1', 'read_capacity': {'mode': 'OnDemand', 'status': {'state': 'Ready', 'current_shards': None,
# 'current_replicas': None, 'error_message': None}}, 'source_collection': None, 'schema':
# {'fields': {'_values': {'type': 'dense_vector', 'dimension': 1536, 'metric': 'euclidean',
# 'description': None}, '_sparse_values': {'type': 'sparse_vector', 'description': None}}}},
# 'pod': None, 'byoc': None}, 'embed': None}]}


# =====================================================
# Create a New Pinecone Index Using Python
# =====================================================

# Define the configuration for the new index.
#
# index_name_3:
#     The name of the index we want to create.
#
# dimension_3:
#     The number of dimensions each dense vector will contain.
#
# metric_3:
#     The similarity or distance metric used when comparing vectors.
#
# In this example, we'll create an index with:
#
# - Name: my-newnew-index
# - Dimension: 1536
# - Metric: euclidean

index_name_3 = "my-newnew-index"
dimension_3 = 1536
metric_3 = "euclidean"


# =====================================================
# Check Whether the Index Already Exists
# =====================================================

# Retrieve the names of all existing indexes.
#
# The list comprehension extracts the name of each index
# returned by pc.list_indexes().
#
# If my-newnew-index already exists:
#
#     Delete the existing index.
#
# Otherwise:
#
#     Print a message indicating that the index doesn't exist.
#
# This allows us to recreate the index using the configuration
# defined above.
#
# WARNING:
# Deleting an index also deletes its stored vectors and data.

if index_name_3 in [index.name for index in pc.list_indexes()]:

    pc.delete_index(index_name_3)

    print(
        f"{index_name_3} successfully deleted"
    )

else:

    print(
        f"{index_name_3} not in index list."
    )


# =====================================================
# Create the New Index
# =====================================================

# Call create_index() to create a new Pinecone index.
#
# Parameters:
#
# name:
#     The name of the index.
#
# dimension:
#     The number of dimensions for the vectors.
#
# metric:
#     The metric used to compare vectors.
#
# spec:
#     The deployment configuration for the index.
#
# ServerlessSpec specifies:
#
# - Cloud provider: AWS
# - Region: us-east-1

pc.create_index(
    name=index_name_3,
    dimension=dimension_3,
    metric=metric_3,
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


# =====================================================
# Verify the New Index
# =====================================================

# Call list_indexes() again to confirm that the new index
# was created successfully.

# print(pc.list_indexes().to_dict())


# =====================================================
# Example Output: Newly Created Index
# =====================================================

# {'name': 'my-newnew-index', 'status': {'ready': True, 'state': 'Ready'}, 'schema': {'fields': {'_values':
# {'type': 'dense_vector', 'dimension': 1536, 'metric': 'euclidean', 'description': None}, '_sparse_values':
# {'type': 'sparse_vector', 'description': None}}}, 'deployment': {'deployment_type': 'managed', 'cloud': 'aws',
# 'region': 'us-east-1', 'environment': 'aped-4627-b74a'}, 'deletion_protection': 'disabled',
# 'host': 'https://my-newnew-index-6rd4bb8.svc.aped-4627-b74a.pinecone.io', 'read_capacity': {'mode': 'OnDemand',
# 'status': {'state': 'Ready', 'current_shards': None, 'current_replicas': None, 'error_message': None}}, 'tags': None,
# 'private_host': None, 'source_collection': None, 'source_backup_id': None, 'cmek_id': None, 'dimension': 1536, 'metric':
# 'euclidean', 'vector_type': 'dense', 'spec': {'serverless': {'cloud': 'aws', 'region': 'us-east-1', 'read_capacity':
# {'mode': 'OnDemand', 'status': {'state': 'Ready', 'current_shards': None, 'current_replicas': None, 'error_message': None}},
# 'source_collection': None, 'schema': {'fields': {'_values': {'type': 'dense_vector', 'dimension': 1536, 'metric': 'euclidean',
# 'description': None}, '_sparse_values': {'type': 'sparse_vector', 'description': None}}}},
# 'pod': None, 'byoc': None}, 'embed': None},