import os

from dotenv import find_dotenv, load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import pandas as pd
import time


# =====================================================
# Embedding Algorithms — Summary and Quick Reference
# =====================================================

# =====================================================
# 1. Introduction to Embedding Algorithms
# =====================================================

# Embedding algorithms are techniques used to convert
# text into numerical vector representations.
#
# These vectors allow machine learning models and
# vector databases to process and compare text.
#
# This is particularly important for semantic search,
# where we want to retrieve information based on meaning
# rather than relying exclusively on exact keyword matches.
#
# Embedding algorithms are a broad topic that could fill
# many lessons.
#
# For now, we'll focus on the practical concepts needed
# to understand text vectorization and implement our
# semantic search application.
#
# Before generating embeddings in Python, we'll explore
# several approaches to representing text as numerical vectors.


# =====================================================
# 2. Traditional NLP Vectorization Techniques
# =====================================================

# Traditional Natural Language Processing (NLP)
# techniques provide relatively straightforward methods
# for converting text into numerical features.
#
# Two common techniques are:
#
# 1. Bag of Words (BoW)
# 2. TF-IDF (Term Frequency-Inverse Document Frequency)
#
# These methods are useful for classical NLP problems,
# including:
#
# - Spam detection
# - Document classification
# - Keyword-based text analysis
#
# However, they have limitations when it comes to
# understanding semantic meaning.


# -----------------------------------------------------
# 2.1 Bag of Words (BoW)
# -----------------------------------------------------

# Bag of Words represents a document based on the
# words it contains.
#
# It generally disregards:
#
# - Word order
# - Sentence structure
# - Contextual meaning
#
# Instead, it creates a vocabulary from the dataset
# and represents each document using word counts
# or binary indicators.
#
# Example:
#
# Document 1: "I love Python"
# Document 2: "I love JavaScript"
#
# Vocabulary:
#
# ["I", "love", "Python", "JavaScript"]
#
# Bag of Words representations:
#
# Document 1: [1, 1, 1, 0]
# Document 2: [1, 1, 0, 1]
#
# Each position corresponds to a word in the vocabulary.
#
# The numbers represent how frequently each word
# appears in the document.
#
# Limitation:
#
# Bag of Words does not understand that two different
# words can have similar meanings.
#
# For example:
#
# "car" and "automobile"
#
# These are treated as separate features, even though
# they have similar meanings.


# -----------------------------------------------------
# 2.2 TF-IDF
# -----------------------------------------------------

# TF-IDF stands for Term Frequency-Inverse
# Document Frequency.
#
# Like Bag of Words, TF-IDF represents text using
# numerical features.
#
# However, TF-IDF also considers how common or
# uncommon a word is across the dataset.
#
# TF-IDF combines two concepts:
#
# 1. Term Frequency (TF)
#
# Measures how frequently a word appears
# within a document.
#
# 2. Inverse Document Frequency (IDF)
#
# Measures how uncommon a word is across
# the collection of documents.
#
# Words that appear frequently in one document
# but rarely across the dataset generally
# receive higher importance.
#
# Example:
#
# Document 1: "Python programming tutorial"
# Document 2: "JavaScript programming tutorial"
# Document 3: "Python machine learning"
#
# Words such as "programming" and "tutorial"
# appear in multiple documents.
#
# TF-IDF adjusts their importance based on how
# frequently they occur across the entire collection.
#
# Advantages:
#
# - Relatively simple to implement.
# - Useful for document classification.
# - Highlights potentially informative terms.
#
# Limitations:
#
# - Does not inherently understand word meaning.
# - Does not capture contextual relationships.
# - Different words with similar meanings remain
#   separate features.
#
# Although Bag of Words and TF-IDF are useful,
# more advanced embedding techniques can capture
# semantic relationships between words.


# =====================================================
# 3. Neural Network-Based Embeddings
# =====================================================

# Neural network-based embedding techniques provide
# a more advanced approach to representing text.
#
# Instead of relying exclusively on word frequencies,
# these models learn numerical representations from
# patterns in language.
#
# Words that appear in similar contexts can develop
# similar vector representations.
#
# This makes them useful for tasks involving
# semantic similarity.
#
# Examples include:
#
# - Word2Vec
# - BERT
# - ELMo
# - Sentence Transformers


# =====================================================
# 4. Word2Vec
# =====================================================

# Word2Vec is a neural network-based technique
# introduced by researchers at Google.
#
# It learns word embeddings by predicting relationships
# between words and their context.
#
# Word2Vec has two primary training approaches:
#
# 1. Continuous Bag of Words (CBOW)
# 2. Skip-gram


# -----------------------------------------------------
# 4.1 Continuous Bag of Words (CBOW)
# -----------------------------------------------------

# CBOW predicts a target word based on its
# surrounding context words.
#
# Example:
#
# "The cat sits on the mat."
#
# Context:
#
# "The", "sits"
#
# Target:
#
# "cat"
#
# The model learns to predict the target word
# using the surrounding words.


# -----------------------------------------------------
# 4.2 Skip-gram
# -----------------------------------------------------

# Skip-gram performs the opposite task.
#
# It predicts surrounding context words using
# a target word.
#
# Example:
#
# Target: "cat"
#
# Context: "The", "sits"
#
# By learning these relationships, Word2Vec
# generates numerical embeddings that capture
# patterns of word usage.


# -----------------------------------------------------
# 4.3 Advantages and Limitations
# -----------------------------------------------------

# Advantages:
#
# - Produces dense word embeddings.
# - Captures useful semantic relationships.
# - More compact than many traditional
#   vocabulary-based representations.
#
# Limitations:
#
# Standard Word2Vec assigns a fixed embedding
# to each word in its vocabulary.
#
# This means the same word generally receives
# the same vector regardless of its context.
#
# For example:
#
# "The bank will not lend me money."
#
# "The river bank is flooded."
#
# Standard Word2Vec does not produce a different
# contextual embedding for "bank" in each sentence.
#
# This limitation motivated the development
# of contextual embedding models.


# =====================================================
# 5. BERT
# =====================================================

# BERT stands for Bidirectional Encoder
# Representations from Transformers.
#
# BERT was developed by researchers at Google.
#
# Unlike traditional word embedding techniques,
# BERT uses the Transformer architecture
# and attention mechanisms.
#
# These mechanisms allow the model to consider
# relationships between words in a sentence.
#
# BERT was pretrained on large text datasets,
# including Wikipedia and BooksCorpus.
#
# It can also be fine-tuned for specific
# downstream NLP tasks.
#
# Examples include:
#
# - Text classification
# - Named entity recognition
# - Question answering
# - Sentiment analysis
#
# One of BERT's major advantages is its ability
# to generate contextual embeddings.


# -----------------------------------------------------
# 5.1 Contextual Embeddings
# -----------------------------------------------------

# Contextual embeddings change depending on
# the surrounding text.
#
# Consider the word "bank" in two sentences.
#
# Example 1: Financial institution
#
# "The bank will not lend me money."
#
# Illustrative embedding:
#
# bank = [0.12, 0.34, -0.56, 0.78, -0.90, ...]
#
# Meaning: Financial institution
#
# Example 2: Riverbank
#
# "The river bank is flooded."
#
# Illustrative embedding:
#
# bank = [0.23, -0.45, 0.67, -0.89, 0.91, ...]
#
# Meaning: Land alongside a river
#
# The same word appears in both sentences,
# but its meaning changes based on context.
#
# BERT can generate different contextual
# representations for these occurrences.
#
# Note:
#
# The numerical values above are illustrative.
# They are not actual outputs from a BERT model.


# -----------------------------------------------------
# 5.2 Advantages and Limitations
# -----------------------------------------------------

# Advantages:
#
# - Captures contextual word meaning.
# - Uses bidirectional context.
# - Supports many NLP applications.
# - Can be fine-tuned for specific tasks.
#
# Limitations:
#
# - More computationally expensive than simpler
#   embedding techniques.
# - Requires more memory and processing power.
# - Standard BERT embeddings are not necessarily
#   optimized for direct sentence similarity.
#
# For semantic search, specialized models such
# as Sentence Transformers are often more convenient.
#
# Important:
#
# BERT is not primarily a machine translation model.
#
# It is a general-purpose language representation
# model used across many NLP tasks.


# =====================================================
# 6. ELMo
# =====================================================

# ELMo stands for Embeddings from Language Models.
#
# ELMo generates contextual word representations
# using deep bidirectional Long Short-Term Memory
# (LSTM) networks.
#
# LSTMs are a type of recurrent neural network
# designed to learn relationships in sequences.
#
# ELMo analyzes surrounding context to generate
# representations that reflect how a word is used
# in a particular sentence.
#
# Like BERT, ELMo produces contextual embeddings.
#
# However:
#
# - ELMo uses bidirectional LSTM networks.
# - BERT uses Transformer-based attention.


# -----------------------------------------------------
# 6.1 Example: Words with Multiple Meanings
# -----------------------------------------------------

# Consider the word "bat".
#
# Example 1: Sports equipment
#
# "The baseball player swung the bat."
#
# Illustrative embedding:
#
# bat = [0.35, 0.20, -0.25, ...]
#
# Meaning: Sports equipment
#
# Example 2: Animal
#
# "The bat flew out of the cave."
#
# Illustrative embedding:
#
# bat = [0.25, -0.10, 0.45, ...]
#
# Meaning: Flying mammal
#
# ELMo can generate different representations
# for "bat" depending on its context.
#
# This is particularly useful for polysemous words,
# which have multiple meanings.
#
# Note:
#
# These vectors are illustrative examples,
# not actual ELMo-generated embeddings.


# -----------------------------------------------------
# 6.2 Advantages and Limitations
# -----------------------------------------------------

# Advantages:
#
# - Generates contextual word representations.
# - Handles words with multiple meanings.
# - Uses information from both directions
#   of a sentence.
#
# Limitations:
#
# - More computationally expensive than
#   traditional vectorization techniques.
# - Uses recurrent networks rather than
#   Transformer-based attention.
# - Is less commonly selected for new
#   semantic search systems today.


# =====================================================
# 7. Sentence Transformers
# =====================================================

# Sentence Transformers are models designed
# to generate meaningful embeddings for sentences,
# paragraphs, and other text.
#
# They are particularly useful for:
#
# - Semantic search
# - Sentence similarity
# - Document retrieval
# - Text clustering
# - Duplicate detection
#
# Unlike traditional word embedding models,
# Sentence Transformers can represent an entire
# text passage as a single vector.
#
# This is especially useful for our Pinecone
# semantic search application.
#
# Instead of comparing individual words,
# we can compare vector representations
# of entire course descriptions.


# -----------------------------------------------------
# 7.1 Example: Semantic Similarity
# -----------------------------------------------------

# Consider the following sentences:
#
# Sentence 1:
#
# "Learn how to analyze data using Python."
#
# Sentence 2:
#
# "Study Python techniques for data analysis."
#
# Sentence 3:
#
# "Learn how to bake a chocolate cake."
#
# A well-trained sentence embedding model should
# generally represent sentences 1 and 2 as more
# semantically similar than sentences 1 and 3.
#
# We can store these embeddings in Pinecone
# and use similarity metrics to retrieve
# relevant course descriptions.


# =====================================================
# 8. Choosing a Sentence Transformer Model
# =====================================================

# The Sentence Transformers library provides
# several pretrained embedding models.
#
# Different models have different trade-offs involving:
#
# - Embedding quality
# - Inference speed
# - Model size
# - Memory usage
# - Vector dimensions
#
# When comparing pretrained models, we may
# encounter benchmark scores for semantic
# search performance.
#
# Higher benchmark scores generally indicate
# better performance on the evaluated tasks.
#
# However, a model with the highest benchmark
# score is not necessarily the best choice
# for every application.
#
# Larger models may provide better retrieval
# quality but require more computational resources.
#
# Smaller models can offer faster inference
# and lower memory requirements.
#
# The best choice depends on the requirements
# of the application.


# -----------------------------------------------------
# 8.1 Our Selected Embedding Model
# -----------------------------------------------------

# For our semantic search application, we'll use:
#
# all-MiniLM-L6-v2
#
# This is a pretrained Sentence Transformer model.
#
# It generates 384-dimensional embeddings.
#
# Advantages:
#
# - Relatively small model size.
# - Fast inference.
# - Lower computational requirements.
# - Suitable for many semantic similarity tasks.
# - Convenient for learning and experimentation.
#
# Although larger models may achieve better
# benchmark results, all-MiniLM-L6-v2 provides
# a useful balance between speed and quality.
#
# This makes it a practical choice for
# our course search project.
#
# Recall our Pinecone index configuration
# from PineconeSetup.py:
#
# index_name = "my-index"
# dimension = 384
# metric = "cosine"
#
# The dimension of our Pinecone index must
# match the output dimension of our embedding model.
#
# Since all-MiniLM-L6-v2 generates
# 384-dimensional vectors, our index
# is configured with dimension=384.


# =====================================================
# 9. Comparing Embedding Techniques
# =====================================================

# Below is a comparison of the techniques
# covered in this lesson.


# -----------------------------------------------------
# Bag of Words
# -----------------------------------------------------

# How it works:
#
# Represents text using word counts
# or binary indicators.
#
# Context awareness:
#
# No.
#
# Common applications:
#
# - Spam detection
# - Document classification


# -----------------------------------------------------
# TF-IDF
# -----------------------------------------------------

# How it works:
#
# Weights words based on term frequency
# and document frequency.
#
# Context awareness:
#
# No.
#
# Common applications:
#
# - Document classification
# - Keyword-based retrieval


# -----------------------------------------------------
# Word2Vec
# -----------------------------------------------------

# How it works:
#
# Learns word embeddings through
# context prediction.
#
# Context awareness:
#
# Limited; fixed vectors.
#
# Common applications:
#
# - Word similarity
# - NLP features


# -----------------------------------------------------
# BERT
# -----------------------------------------------------

# How it works:
#
# Uses Transformer attention to generate
# contextual token embeddings.
#
# Context awareness:
#
# Yes.
#
# Common applications:
#
# - Classification
# - Question answering
# - Named entity recognition


# -----------------------------------------------------
# ELMo
# -----------------------------------------------------

# How it works:
#
# Uses bidirectional LSTMs to generate
# contextual word embeddings.
#
# Context awareness:
#
# Yes.
#
# Common applications:
#
# - Contextual word representation
# - NLP tasks


# -----------------------------------------------------
# Sentence Transformers
# -----------------------------------------------------

# How it works:
#
# Generates embeddings optimized for
# comparing text passages.
#
# Context awareness:
#
# Yes.
#
# Common applications:
#
# - Semantic search
# - Clustering
# - Similarity


# -----------------------------------------------------
# 9.1 Key Differences
# -----------------------------------------------------

# Traditional NLP techniques:
#
# - Focus on word occurrence and frequency.
# - Are relatively simple and computationally efficient.
# - Do not inherently understand semantic meaning.
#
# Neural network-based techniques:
#
# - Learn representations from language patterns.
# - Can capture semantic relationships.
# - Often require more computational resources.
#
# Contextual embedding models:
#
# - Generate representations based on surrounding text.
# - Can distinguish between different meanings
#   of the same word.
#
# Sentence Transformers:
#
# - Generate embeddings for complete text passages.
# - Are particularly useful for semantic
#   similarity and retrieval.


# =====================================================
# 10. How Embeddings Fit Into Our Pinecone Project
# =====================================================

# Our goal is to improve the search functionality
# of the 365 Data Science platform.
#
# Instead of relying exclusively on keyword matching,
# we want to retrieve courses based on semantic similarity.
#
# Our workflow will be:


# -----------------------------------------------------
# Step 1: Load Course Descriptions
# -----------------------------------------------------

# Read the course dataset from:
#
# course_descriptions.csv


# -----------------------------------------------------
# Step 2: Preprocess the Data
# -----------------------------------------------------

# Combine relevant course information
# into a single text field.
#
# For example:
#
# files["new_course_description"]


# -----------------------------------------------------
# Step 3: Generate Embeddings
# -----------------------------------------------------

# Use all-MiniLM-L6-v2 to convert each
# course description into a 384-dimensional vector.


# -----------------------------------------------------
# Step 4: Store Embeddings in Pinecone
# -----------------------------------------------------

# Upload the generated vectors
# to our Pinecone index.


# -----------------------------------------------------
# Step 5: Process a Search Query
# -----------------------------------------------------

# Convert the user's search query into
# an embedding using the same model.


# -----------------------------------------------------
# Step 6: Perform Semantic Search
# -----------------------------------------------------

# Compare the query embedding with the
# stored course embeddings using cosine similarity.


# -----------------------------------------------------
# Step 7: Retrieve Relevant Courses
# -----------------------------------------------------

# Return courses whose embeddings are
# most similar to the search query.


# -----------------------------------------------------
# 10.1 Workflow Summary
# -----------------------------------------------------

# Course Descriptions
#         |
#         v
# Data Preprocessing
#         |
#         v
# Sentence Transformer
# (all-MiniLM-L6-v2)
#         |
#         v
# 384-Dimensional Embeddings
#         |
#         v
# Pinecone Vector Database
#         |
#         v
# Semantic Similarity Search
#         |
#         v
# Relevant Course Results


# =====================================================
# 11. Summary and Key Takeaways
# =====================================================

# In this lesson, we explored several techniques
# for converting text into numerical vectors.
#
# We began with traditional NLP approaches:
#
# - Bag of Words
# - TF-IDF
#
# We then explored neural network-based
# embedding techniques:
#
# - Word2Vec
# - BERT
# - ELMo
#
# Finally, we introduced Sentence Transformers,
# which are particularly useful for generating
# embeddings for semantic search applications.


# -----------------------------------------------------
# 11.1 Key Takeaways
# -----------------------------------------------------

# 1. Bag of Words
#
# Represents text using word counts or binary
# indicators but does not inherently capture meaning.
#
# 2. TF-IDF
#
# Improves upon basic word counts by considering
# how common or uncommon words are across documents.
#
# 3. Word2Vec
#
# Learns dense word embeddings but assigns
# a fixed representation to each word.
#
# 4. BERT
#
# Uses Transformer architecture to generate
# contextual token embeddings.
#
# 5. ELMo
#
# Uses bidirectional LSTM networks to generate
# contextual word embeddings.
#
# 6. Sentence Transformers
#
# Generate embeddings for complete text passages
# and are useful for semantic search.
#
# 7. all-MiniLM-L6-v2
#
# Generates 384-dimensional embeddings and
# provides a useful balance between speed
# and performance.
#
# 8. Pinecone
#
# Stores our embeddings and allows us
# to perform similarity searches.


# -----------------------------------------------------
# 11.2 Next Steps
# -----------------------------------------------------

# In the next practical lesson, we'll use
# all-MiniLM-L6-v2 to convert our course
# descriptions into embeddings.
#
# We'll then store those embeddings in Pinecone
# and use them to perform semantic searches.
#
# The ultimate goal is to build a search system
# that retrieves relevant courses based on meaning
# rather than exact keyword matches.

# We already discussed the different types of embedding algorithms and what they're most
# suited for, but at the early stages of our implementation, we saw that changing the algoithm didn't
# drastically improve results because the underlying data wasn't granular enough.

# Now that we've added another level by including section information, it's worth exploring the embedding
# options available once again.

# Choosing multi-qa-distilbert-cos-v1, which has 768 dimensions, twice as many as our original index.
# Remember, specifying the right dimensions count is essential when creating our index, but crucially,
# this model has been trained to perform semantic search, which is, after all, precisely what we're
# trying to achieve.


# =====================================================
# 12. Loading Environment Variables
# =====================================================

# Load the Pinecone API key from our .env file.
#
# find_dotenv() searches for the environment file.
#
# override=True allows values from the .env file
# to replace existing environment variables
# with the same names.

load_dotenv(find_dotenv(), override=True)

api_key = os.environ.get("PINECONE_API_KEY")


# Check whether the API key was successfully loaded.
#
# If the API key is missing, raise an error
# explaining why the application cannot continue.

if not api_key:
    raise ValueError(
        "PINECONE_API_KEY was not found in the environment."
    )


# =====================================================
# 13. Initializing the Pinecone Client
# =====================================================

# Initialize the Pinecone client using our API key.
#
# The client allows us to connect to our existing
# vector database and perform similarity searches.

pc = Pinecone(
    api_key=api_key
)


# =====================================================
# 14. Choosing a Separate Index for the New Model
# =====================================================

# Earlier, UpdatingVectorDatabase.py used the
# all-MiniLM-L6-v2 model (384 dimensions) with "my-index".
# Those earlier notes remain above as a learning reference.
#
# In this experiment we use multi-qa-distilbert-cos-v1,
# which produces 768-dimensional embeddings.
# We store them in a separate Pinecone index named "bert".
# The dimension of the index must match the model output.
#
# Each stored vector represents one course section and has:
# - A course-section ID
# - Its numerical embedding
# - Course and section metadata
#
# Unlike a search-only script, this experiment generates
# dataset embeddings and upserts them into "bert".
# Running the script again will update vectors with the
# same IDs, but it will not delete the entire index.
#
# If you only want to search existing vectors without
# regenerating dataset embeddings, set UPLOAD_EMBEDDINGS
# to False after the initial upload.

index_name = "bert"
dimension = 768
metric = "cosine"
UPLOAD_EMBEDDINGS = True  # Set to False after the initial upload to search only.


# =====================================================
# 15. Loading the New Embedding Model
# =====================================================

# Use the same model for dataset vectors and search queries.
# Mixing embeddings from different models, even if they have
# matching dimensions, can produce unreliable similarity scores.

model = SentenceTransformer("multi-qa-distilbert-cos-v1")

# =====================================================
# 16. Creating or Connecting to the "bert" Index
# =====================================================

# Create the index only if it does not exist.
# Never delete an existing index automatically.

existing_indexes = [item.name for item in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created Pinecone index: {index_name}")
else:
    print(f"Using existing Pinecone index: {index_name}")

# Verify that the existing index is compatible with our model.
index_description = pc.describe_index(index_name)
if index_description.dimension != dimension or index_description.metric != metric:
    raise ValueError(
        f"Index '{index_name}' must use {dimension} dimensions and {metric} similarity."
    )

# A newly created serverless index may take time to become ready.
while not pc.describe_index(index_name).status["ready"]:
    print("Waiting for Pinecone index to become ready...")
    time.sleep(2)

index = pc.Index(index_name)


# =====================================================
# 17. Generating and Uploading Course-Section Embeddings
# =====================================================

if UPLOAD_EMBEDDINGS:
    # Read the course-section dataset and build unique vector IDs.
    files = pd.read_csv(
        "course_section_descriptions.csv",
        encoding="cp1252"
    )

    files["unique_id"] = (
        files["course_id"].astype(str) + "-" + files["section_id"].astype(str)
    )

    # Keep course and section details as searchable metadata.
    files["metadata"] = files.apply(
        lambda row: {
            "course_name": row["course_name"],
            "section_name": row["section_name"],
            "section_description": row["section_description"]
        },
        axis=1
    )

    def create_embeddings(row):
        # Combine course and section information before encoding.
        # The model converts this text into a 768-dimensional vector.
        # Suppress per-row progress bars for cleaner output.
        combined_text = f"""{row['course_name']} {row['course_technology']}
{row['course_description']} {row['section_name']} {row['section_description']}
"""
        return model.encode(combined_text, show_progress_bar=False)

    files["embedding"] = files.apply(create_embeddings, axis=1)

    # Upsert updates existing vector IDs or inserts new ones.
    # It does not delete other vectors already in the index.
    vectors_to_upsert = [
        (row["unique_id"], row["embedding"].tolist(), row["metadata"])
        for _, row in files.iterrows()
    ]

    # Upload in batches to avoid overly large requests.
    batch_size = 100
    for start in range(0, len(vectors_to_upsert), batch_size):
        index.upsert(vectors=vectors_to_upsert[start:start + batch_size])

    print("Data successfully upserted to Pinecone index.")
else:
    print("Skipping dataset embeddings; searching existing vectors.")


# =====================================================
# 18. Searching the New Index
# =====================================================

query = "regression in Python"

query_embedding = model.encode(query, show_progress_bar=False).tolist()

query_results = index.query(
    vector=query_embedding,
    top_k=12,
    include_values=False,
    include_metadata=True
)

score_threshold = 0.4

# Assuming query_results are fetched and include metadata

shown_matches = 0

for match in query_results['matches']:
    if match['score'] >= score_threshold:
        shown_matches += 1
        course_details = match.get("metadata", {})
        course_name = course_details.get("course_name", 'N/A')
        section_name = course_details.get("section_name", 'N/A')
        section_description = course_details.get('section_description', 'No description available')

        print(f"Match item ID: {match['id']}, Score: {match['score']}")
        print(f"Course: {course_name} \nSection: {section_name} \nDescription: {section_description}")

if shown_matches == 0:
    print("No relevant course sections found above the similarity threshold.")



# Running our regression query

# The Machine Learning with Ridge and Lasso Regression course is the most relevant result, because the entire
# course is dedicated to implementing different regression techniques in Python.

# Although the word Python isn't specified in the course or section titles, it's mentioned 
# in the description.

# Despite this section focusing on programming prerequisites, it remains integral to the course
# which centers on Python regression.

# I'm thrilled to see sections from the same course among the results, including the regression
# chapter from the KNN course, which is wonderful.

# We also get results from the Customer Analytics in Python course, where we apply multi-class
# linear regression and logistic regression to predict price elasticity for different foods.

# An intro to Python section has snuck in somehow, but overall the results are outstanding.

# But our previous searches revealed none of that.

# We've presented our various parameters in text, combined them, and converted them into vector 
# representations.

# Consider the following scenario.

# I know the course name holds more significance than the technology used, yet our semantic search
# engine currently overlooks this.

# That's why our task will be to introduce weights for the course name, section name, section
# description or any other feature deemed necessary.

# You'll then experiment with different weights, embedding functions, and any other changes you wish
# to implement to make your search algorithm superior.

# I've included an example code for the weighted version of the search, but ultimately
# the best approach is up to you.

# Match item ID: 37-369, Score: 0.676096916
# Course: Machine Learning in Python 
# Section: Linear Regression with sklearn 
# Description: While there are many libraries that can compute a regression model, the most numerically stable one is sklearn. It is also the preferred choice of many machine learning professionals. In this section, we implement all we know about regressions in this amazing library.
# Match item ID: 37-368, Score: 0.606089592
# Course: Machine Learning in Python 
# Section: Linear Regression 
# Description: In this part of the course, we will discuss what the course covers, why you need to learn advanced statistics, what’s the differences are with machine learning, and how to get the most out of this training. In this section, you will also expand on what you learned in our statistics training with additional concepts and will apply all the theory in Python. This section serves two purposes: 1) a useful refresher of regression, and 2) a great way to reinforce what you have learned, applying it in practice while coding.
# Match item ID: 36-363, Score: 0.563509941
# Course: Python for Finance 
# Section: Using Regressions for Financial Analysis 
# Description: Understanding rates of return and risk is not all there is about finance. Working with regression analysis is a must, and you will see that Python only helps you to be quicker and more precise when doing such estimations.
# Match item ID: 56-491, Score: 0.558063507
# Course: Machine Learning with Ridge and Lasso Regression 
# Section: Setting Up The Environment 
# Description: If you’re new to programming with Python, we recommend going through our Introduction to Jupyter course which details installing Anaconda and Jupyter and features a tour of the Jupyter Environment. Here, we talk about the required packages for applying ridge and lasso regression in Python. 
# Match item ID: 36-366, Score: 0.555825293
# Course: Python for Finance 
# Section: Multivariate Regression Analysis 
# Description: While in Section 4 we deal with simple regression analysis, here we will take this technique to the next level. As you can guess, multivariate regression analysis is more advanced, but is also more interesting, as it allows you to deal with more complex financial problems.
# Match item ID: 37-370, Score: 0.494563103
# Course: Machine Learning in Python 
# Section: Linear Regression Practical Example 
# Description: An all in one practical example, which will test your understanding of each of the concepts that we have discussed so far. We will focus on a used cars dataset and create a linear regression model to predict the prices of cars. At the end you will have a big assignment where you can dive deep into the optimization of a machine learning model.
# Match item ID: 37-371, Score: 0.486801147
# Course: Machine Learning in Python 
# Section: Logistic Regression 
# Description: Data scientists use logistic regressions when the dependent variable is binary (0 and 1, true and false, etc.). This type of data is encountered on a daily basis when working as a data scientist, and here, you will learn how to build a logistic regression, understand tables, interpret the coefficients of a logistic regression, calculate the accuracy of the model, as well as how to test. We will introduce under and overfitting, and will teach you how to test your models.
# Match item ID: 55-488, Score: 0.468299925
# Course: Machine Learning with K-Nearest Neighbors 
# Section: KNN Regressor 
# Description: Continuing the practical part of the course, we will dive into solving regression tasks using the K-Nearest Neighbors method. Similar to what we did in the previous section, we will tackle this problem by generating 2 random datasets. One would represent a linear problem, while the other would be non-linear. We will apply a linear (parametric) model and a KNN (non-parametric model) on both datasets and argue which one performs better. 
# Match item ID: 33-341, Score: 0.461848259
# Course: Customer Analytics in Python 
# Section: Modeling Purchase Quantity 
# Description: We conclude the purchase analytics part of the course with a model for purchase quantity. Here, we’ll use linear regression to determine how many units of our product the customer likes to buy. We’ll examine the effects of promotion on purchase quantity and determine price elasticity of demand.
# Match item ID: 35-357, Score: 0.449158669
# Course: Introduction to Python 
# Section: Functions 
# Description: Python functions are another invaluable tool for programmers. They allow you to carry out pre-defined or specifically-designed operations that manipulate the data you are working with and bring it one step closer to representing a meaningful output.
# Match item ID: 56-492, Score: 0.449085265
# Course: Machine Learning with Ridge and Lasso Regression 
# Section: Ridge and Lasso Regression – Practical Case 
# Description: In this section, we will walk you through the implementation of ridge and lasso regression using sk-learn in Python. We apply these methods to a real dataset in order to increase the performance of a regression algorithm by preventing overfitting. Furthermore, we demonstrate how regularization works and uncover the differences between ridge and lasso models.  
# Match item ID: 56-490, Score: 0.440646172
# Course: Machine Learning with Ridge and Lasso Regression 
# Section: Regularization Basics 
# Description: As an introduction to the course, we explore the concept of regularization and explain how it can be leveraged to prevent overfitting and multicollinearity issues. In addition, we demonstrate the theoretical differences between the mechanisms of ridge and lasso regression. 


# What other problems can be solved with vector databases.

# I spent a good chunk of the course talking about this specific topic, because i wanted to explain 
# each step in the process in more detail, to give you a complete understanding of how vector 
# databases work.

# With that knowledge, it'll be easier to solve other problems and implement your own algoriths,
# rather than simply treating vector stores as black boxes.