import os

from dotenv import find_dotenv, load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. Introduction to Similarity Search
# =====================================================

# Our course-section embeddings are already stored
# in Pinecone.
#
# However, the search options available directly
# through the Pinecone dashboard are limited.
#
# In this lesson, we'll use Python to perform
# semantic searches against our existing vector database.
#
# Semantic search allows us to retrieve information
# based on meaning rather than exact keyword matches.
#
# Our goal is to identify course sections that are
# semantically similar to a user's search query.


# =====================================================
# 2. Loading Environment Variables
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
# 3. Initializing the Pinecone Client
# =====================================================

# Initialize the Pinecone client using our API key.
#
# The client allows us to connect to our existing
# vector database and perform similarity searches.

pc = Pinecone(
    api_key=api_key
)


# =====================================================
# 4. Connecting to Our Existing Pinecone Index
# =====================================================

# We'll use the same index created in
# UpdatingVectorDatabase.py.
#
# Index name: my-index
#
# Our index contains 384-dimensional embeddings
# representing individual course sections.
#
# Each vector has:
#
# - A unique course-section ID
# - Numerical embedding values
# - Metadata describing the course and section
#
# We don't need to recreate the index or
# upload the embeddings again.
#
# We'll simply connect to the existing index.
#
# Dataset embeddings should not be regenerated
# for every search. They are already stored
# in Pinecone, so repeating that work adds
# unnecessary computation and indexing work.
#
# Regenerate affected dataset embeddings when
# the source content or embedding setup changes,
# using the separate indexing workflow.
#
# For each new search, only the query needs
# to be converted into an embedding.

index_name = "my-index"

index = pc.Index(index_name)


# =====================================================
# 5. Loading the Embedding Model
# =====================================================

# We'll use the same Sentence Transformers model
# that generated our course-section embeddings.
#
# Model: all-MiniLM-L6-v2
#
# This model converts text into 384-dimensional
# numerical vectors.
#
# Using the same embedding model is important
# because our query must be represented in
# the same vector space as our stored data.
#
# If we used an incompatible embedding model,
# the similarity scores might not be meaningful.

model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# 6. Defining Our Search Query
# =====================================================

# First, we'll define the search query.
#
# Previously, we searched for topics related
# to clustering.
#
# Now, we'll search for regression courses
# specifically related to Python.
#
# Adding "in Python" provides additional context
# about the technology we're interested in.
#
# Our goal is to retrieve relevant Python
# course sections covering regression.
#
# Note:
#
# Semantic search doesn't guarantee that every
# result will exclusively cover Python.
#
# The model ranks results according to
# semantic similarity.

# Query wording changes the query embedding
# and can change both rankings and scores.
#
# For example, "regression" and "regression in Python"
# may retrieve different sections because Python
# adds a programming-language context.
#
# Try wording that clearly describes the topic
# and context you need, then review the results.
# Adding words does not guarantee better results.

query = "regression in Python"


# =====================================================
# 7. Creating the Query Embedding
# =====================================================

# Our course-section information is stored
# as numerical vectors.
#
# Therefore, we must convert the search query
# into a numerical vector before searching.
#
# We'll use model.encode() to generate
# the query embedding.
#
# show_progress_bar=False prevents the model
# from displaying a progress bar.
#
# The resulting embedding is a NumPy array.
#
# We'll use tolist() to convert it into
# a standard Python list.
#
# Pinecone expects a flat list of numerical
# values when querying with a single vector.

query_embedding = model.encode(
    query,
    show_progress_bar=False
).tolist()


# =====================================================
# 8. Performing the Similarity Search
# =====================================================

# Now we'll use Pinecone's query() method
# to search our vector database.
#
# The query() method compares our query embedding
# with the vectors stored in the index.
#
# We'll use the following parameters:
#
# vector:
#
# The numerical representation of our search query.
#
# top_k:
#
# Specifies the maximum number of matches
# returned by Pinecone.
#
# Setting top_k=12 requests the 12 most similar
# vectors from our database.
#
# include_values:
#
# Determines whether Pinecone returns the
# numerical embedding values of each match.
#
# Setting include_values=False means we don't
# need to retrieve the 384 numerical values.
#
# include_metadata:
#
# Determines whether Pinecone returns
# the metadata associated with each vector.
#
# Setting include_metadata=True allows us
# to retrieve information such as:
#
# - Course name
# - Section name
# - Section description
#
# This is especially important because
# numerical embeddings aren't human-readable.
#
# The metadata helps us understand which
# course sections were retrieved.

query_results = index.query(
    vector=query_embedding,
    top_k=12,
    include_values=False,
    include_metadata=True
)


# =====================================================
# 9. Understanding the Query Results
# =====================================================

# Pinecone returns the search results
# inside the "matches" collection.
#
# Each match represents a stored vector
# that is similar to our query embedding.
#
# A match can contain:
#
# - id: The unique identifier of the vector
# - score: The similarity score
# - values: The numerical vector, if requested
# - metadata: Additional information, if requested
#
# Our course-section IDs combine the original
# course ID and section ID.
#
# For example:
#
# 37-369
#
# This identifies a specific section
# belonging to a particular course.
#
# Because include_metadata=True,
# we can also retrieve the corresponding
# course name, section name, and description.
#
# Since include_values=False, the numerical
# embedding values aren't included.


# =====================================================
# 10. Filtering Results With a Similarity Threshold
# =====================================================

# Not every result returned by Pinecone
# will necessarily be relevant to our query.
#
# One way to filter the results is by
# applying a similarity threshold.
#
# A similarity threshold specifies the
# minimum score required for a result
# to be displayed.
#
# For this example, we'll use 0.4.
#
# This means:
#
# - Scores >= 0.4 will be displayed.
# - Scores < 0.4 will be excluded.
#
# Increasing the threshold can remove
# lower-scoring results.
#
# However, a higher threshold may also
# exclude potentially relevant matches.
#
# Conversely, lowering the threshold
# may return more results, but it can
# increase the number of irrelevant matches.
#
# IMPORTANT:
#
# A similarity threshold doesn't improve
# the underlying embedding model.
#
# It only filters the results according
# to their similarity scores.
#
# top_k and the threshold serve different purposes:
#
# top_k=12 requests up to 12 matches from Pinecone.
# The threshold then filters those returned matches
# locally, so fewer than 12 may be displayed.
#
# Lowering the threshold does not retrieve matches
# outside those 12. Increasing top_k requests more
# candidates, but does not make them more relevant.


# =====================================================
# 11. Displaying Search Results and Metadata
# =====================================================

# We'll iterate through the matches returned
# by Pinecone using a for loop.
#
# For each match, we'll check whether
# its similarity score is at least 0.4.
#
# If the score meets our threshold,
# we'll retrieve the associated metadata.
#
# We'll use the dictionary get() method
# to safely access metadata values.
#
# The get() method accepts:
#
# - The key we want to retrieve
# - A fallback value if the key is missing
#
# For example:
#
# course_details.get("section_name", "N/A")
#
# If section_name is missing, the result
# will display "N/A" instead of raising
# a KeyError.
#
# We'll use the following fallback values:
#
# - course_name: "N/A"
# - section_name: "N/A"
# - section_description: "No description"
#
# Finally, we'll display:
#
# - Matched item ID
# - Similarity score
# - Course name
# - Section name
# - Section description
#
# Displaying the descriptions helps us
# evaluate whether the retrieved sections
# are actually relevant to our query.

# Track whether any returned match passes
# the threshold, including an empty matches list.

results_found = False

for match in query_results["matches"]:

    # Only display results with a similarity
    # score greater than or equal to 0.4.

    if match["score"] >= 0.4:

        results_found = True

        # Retrieve the metadata dictionary.
        #
        # Use an empty dictionary as a fallback
        # if metadata is missing.

        course_details = match.get("metadata") or {}

        # Retrieve individual metadata values.
        #
        # Provide fallback values in case
        # any information is unavailable.

        course_name = course_details.get(
            "course_name",
            "N/A"
        )

        section_name = course_details.get(
            "section_name",
            "N/A"
        )

        section_description = course_details.get(
            "section_description",
            "No description"
        )

        # Display the matched vector ID
        # and its similarity score.

        print(
            f"Matched item ID: {match['id']}, "
            f"Score: {match['score']}"
        )

        # Display the course-section metadata.
        #
        # The \n characters insert line breaks
        # between the displayed values.

        print(
            f"Course: {course_name}\n"
            f"Section: {section_name}\n"
            f"Description: {section_description}"
        )


# Explain why no results were displayed
# if none of the returned matches met the threshold.

if not results_found:
    print("No results found with a similarity score of at least 0.4.")


# =====================================================
# 12. Evaluating Our Clustering Search Results
# =====================================================

# In our earlier experiment, we searched
# for course sections related to clustering.
#
# Several relevant sections appeared
# near the top of our results.
#
# For example:
#
# Machine Learning in Excel
# Section: Cluster Analysis
# Score: 0.561096907
#
# Machine Learning in Python
# Section: Other Types of Clustering
# Score: 0.543184936
#
# Machine Learning in Excel
# Section: K-means Clustering
# Score: 0.509194136
#
# Customer Analytics in Python
# Section: Hierarchical Clustering
# Score: 0.506996393
#
# Customer Analytics in Python
# Section: K-means Clustering
# Score: 0.490018487
#
# These results demonstrate how section-level
# embeddings can identify specific topics
# within larger courses.
#
# For example, K-means clustering appears
# in several different courses.
#
# By generating embeddings for individual
# sections, we can retrieve these topics
# without relying exclusively on course titles.
#
# However, some results were less relevant.
#
# For example:
#
# Machine Learning in Excel
# Section: Introduction
# Score: 0.40000084
#
# The introductory section doesn't directly
# focus on clustering.
#
# Another result involved decision trees.
#
# Although decision trees aren't primarily
# clustering algorithms, they involve
# splitting data into different branches.
#
# This may explain why the embedding model
# considers them semantically related.
#
# We also retrieved sections from
# Fashion Analytics with Tableau.
#
# One section discussed consumer clusters,
# making it somewhat relevant even though
# clustering wasn't the main course topic.
#
# Overall, our clustering results included
# several useful recommendations.
#
# However, the presence of less relevant
# sections demonstrates that semantic
# similarity doesn't guarantee relevance.


# =====================================================
# 13. Evaluating Our Regression Search Results
# =====================================================

# Next, we changed our query to:
#
# "regression in Python"
#
# We also changed the similarity threshold
# to 0.4.
#
# Adding Python to our query helps communicate
# that we're interested in regression topics
# using that programming language.
#
# In our earlier results, several relevant
# sections appeared near the top.
#
# For example:
#
# Machine Learning in Python
# Section: Linear Regression with sklearn
# Score: 0.752672911
#
# Python for Finance
# Section: Using Regressions for Financial Analysis
# Score: 0.675911546
#
# Machine Learning in Python
# Section: Linear Regression
# Score: 0.636868775
#
# Python for Finance
# Section: Multivariate Regression Analysis
# Score: 0.576831043
#
# Machine Learning in Python
# Section: Linear Regression Practical Example
# Score: 0.560969234
#
# These results are closely related
# to our search query.
#
# However, some less relevant results
# also appeared.
#
# For example:
#
# Introduction to Python
# Section: Functions
# Score: 0.553500056
#
# Introduction to Python
# Section: Python Variables and Data Types
# Score: 0.544553161
#
# Introduction to Python
# Section: Basic Python Syntax
# Score: 0.534280777
#
# Although these sections involve Python,
# they don't directly focus on regression.
#
# This demonstrates an important limitation
# of our current embedding approach.
#
# The model may consider the shared
# programming language relevant even when
# the specific topic doesn't match.
#
# One of our earlier results also included
# a course-level vector:
#
# ID: machine-learning-in-python
#
# This vector didn't contain the same
# section-level metadata as our newer entries.
#
# It appeared because our earlier index
# contained a mixture of course-level
# and section-level embeddings.
#
# Our updated vector database should now
# contain the course-section embeddings
# created in UpdatingVectorDatabase.py.
#
# This allows us to evaluate the results
# using consistent section-level metadata.


# =====================================================
# 14. Understanding Similarity Scores
# =====================================================

# The similarity score indicates how closely
# the query vector matches a stored vector.
#
# Our Pinecone index uses cosine similarity.
#
# Cosine similarity compares the direction
# of vectors in a multidimensional space.
#
# Generally, higher scores indicate
# greater semantic similarity.
#
# However, similarity scores aren't
# probabilities of correctness.
#
# For example:
#
# A score of 0.75 doesn't mean that
# a result is 75% accurate.
#
# It means the vectors have a cosine
# similarity score of 0.75.
#
# A high similarity score doesn't
# guarantee that a result is useful
# or directly answers the query.
#
# This is why reviewing the section
# descriptions is important.


# =====================================================
# 15. Summary
# =====================================================

# In this lesson, we implemented semantic
# search using Pinecone's query() method.
#
# We learned how to:
#
# - Connect to an existing Pinecone index.
# - Load the same embedding model used for indexing.
# - Convert a search query into an embedding.
# - Search Pinecone for similar vectors.
# - Limit the number of results using top_k.
# - Retrieve similarity scores and metadata.
# - Understand the structure of query results.
# - Display course and section information.
# - Evaluate the relevance of retrieved sections.
# - Filter results using a similarity threshold.
#
# Our current semantic search implementation works,
# but some results may be less relevant than expected.
#
# Applying a similarity threshold can remove
# lower-scoring matches, but it cannot fix
# inaccurate similarity rankings.
#
# The quality of our search results depends on:
#
# - The embedding model
# - The quality of the source data
# - The information included in each embedding
# - The wording of the search query
#
# In future lessons, we'll explore whether
# a different embedding model can improve
# our semantic search results.