import os

from dotenv import find_dotenv, load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pprint


files = pd.read_csv(
    "course_section_descriptions.csv",
    encoding="cp1252"
)

# Assuming 'files' DataFrame is already defined and includes necessary columns

# Define weights for different text components

weight_course_name = 5
weight_section_name = 3
weight_section_description = 2
weight_other = 1 # For other components like course technology and course description

# Create a unique identifier for each section combining course_id and section_id

files['unique_id'] = files['course_id'].astype(str) + "-" + files['section_id'].astype(str)

# Create metadata for each section

files['metadata'] = files.apply(lambda row: {
    "course_name": row['course_name'],
    "section_name": row['section_name'],
    "section_description": row['section_description']
}, axis=1)

def create_embeddings(row):
    # Encode individual components

    emb_course_name = model.encode(row['course_name'], show_progress_bar=False) * weight_course_name
    emb_section_name = model.encode(row['section_name'], show_progress_bar=False) * weight_section_name
    emb_section_description = model.encode(row['section_description'], show_progress_bar=False) * weight_section_description
    emb_course_tech = model.encode(row['course_technology'], show_progress_bar=False) * weight_other
    emb_course_desc = model.encode(row['course_description'], show_progress_bar=False) * weight_other

    # Combine embeddings by averaging them

    combined_embedding = (emb_course_name + emb_section_name + emb_section_description + emb_course_tech + emb_course_desc) / (weight_course_name + weight_section_name + weight_section_description + 2 * weight_other)
    return combined_embedding


# Initialize the model

model = SentenceTransformer('multi-qa-distilbert-cos-v1')

# Apply the function to create weighted embeddings

files['embedding'] = files.apply(create_embeddings, axis=1)

# Connect to Pinecone Index

load_dotenv(find_dotenv(), override=True)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "weighted-features-768"

dimension = model.get_embedding_dimension()

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Prepare the vectors for upserting

vectors_to_upsert = [(row['unique_id'], row['embedding'].tolist(), row['metadata']) for index, row in files.iterrows()]

# Upsert data

index.upsert(vectors=vectors_to_upsert)

print("Data successfully upserted to Pinecone index.")


# Ensure you've already initialized and configured Pinecone and the model
# If not, you need to run the initialization code provided earlier

# Create the query embedding

query = 'lasso'
query_embedding = model.encode(query, show_progress_bar=False).tolist()

query_results = index.query(
    vector=query_embedding,
    top_k=15,
    include_metadata=True
)

score_threshold = 0.3

# Assuming query_results are fetched and include metadata

for match in query_results['matches']:
    if match['score'] >= score_threshold:
        course_details = match.get('metadata', {})
        course_name = course_details.get('course_name', 'N/A')
        section_name = course_details.get('section_name', 'N/A')
        section_description = course_details.get('section_description', 'No description available')

        pprint.pprint(f"Matched item ID: {match['id']}, Score: {match['score']}")
        pprint.pprint(f"Course: {course_name}")
        pprint.pprint(f"Section: {section_name}, Description: {section_description}", width=100)

#         'Matched item ID: 56-492, Score: 0.618949'
# 'Course: Machine Learning with Ridge and Lasso Regression'
# ('Section: Ridge and Lasso Regression – Practical Case, Description: In this section, we will walk '
#  'you through the implementation of ridge and lasso regression using sk-learn in Python. We apply '
#  'these methods to a real dataset in order to increase the performance of a regression algorithm '
#  'by preventing overfitting. Furthermore, we demonstrate how regularization works and uncover the '
#  'differences between ridge and lasso models.  ')
# 'Matched item ID: 56-490, Score: 0.573801458'
# 'Course: Machine Learning with Ridge and Lasso Regression'
# ('Section: Regularization Basics, Description: As an introduction to the course, we explore the '
#  'concept of regularization and explain how it can be leveraged to prevent overfitting and '
#  'multicollinearity issues. In addition, we demonstrate the theoretical differences between the '
#  'mechanisms of ridge and lasso regression. ')
# 'Matched item ID: 56-491, Score: 0.511207104'
# 'Course: Machine Learning with Ridge and Lasso Regression'
# ('Section: Setting Up The Environment, Description: If you’re new to programming with Python, we '
#  'recommend going through our Introduction to Jupyter course which details installing Anaconda and '
#  'Jupyter and features a tour of the Jupyter Environment. Here, we talk about the required '
#  'packages for applying ridge and lasso regression in Python. ')
# 'Matched item ID: 101-705, Score: 0.378146678'
# 'Course: The Machine Learning Algorithms A-Z'
# ('Section: Ridge, Lasso, Elastic Net, Description: One way to limit some of the negatives of '
#  'linear regression is via regularization. We can regularize a linear regression model using '
#  'ridge, lasso, and elastic net algorithms. ')
# 'Matched item ID: 49-459, Score: 0.301573038'
# 'Course: Linear Algebra and Feature Selection'
# ('Section: Principal Component Analysis (PCA), Description: In this part of the course, we explore '
#  'the Principal Component Analysis (PCA) - one of the most widely used algorithms for '
#  "dimensionality reduction. We'll demonstrate a practical example combining both feature "
#  'extraction and feature selection techniques to achieve the desired goal - reducing the number of '
#  'dimensions in our dataset.')