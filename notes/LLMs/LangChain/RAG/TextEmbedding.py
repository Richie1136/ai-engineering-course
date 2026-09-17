from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv

import os
import numpy as np  # Used for numerical arrays and mathematical operations


# =====================================================
# Overview
# =====================================================

# In the practical part of this section, we learned how to load and split data.
#
# The next step is embedding.
#
# Embedding refers to representing a chunk of text as a vector.
#
# This allows each chunk to exist in a vector space where we can measure
# the semantic similarity between vectors.


# =====================================================
# Load the DOCX Document
# =====================================================

loader_docx = Docx2txtLoader(
    "Introduction_to_Data_and_Data_Science_2.docx"
)

pages = loader_docx.load()


# =====================================================
# Split the Document by Markdown Headers
# =====================================================

# Initialize a MarkdownHeaderTextSplitter and use it to split the document
# into individual lectures.

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Course Title"),
        ("##", "Lecture Title")
    ]
)

pages_md_split = md_splitter.split_text(
    pages[0].page_content
)


# =====================================================
# Remove Newline Characters
# =====================================================

# Remove newline characters that clutter the text.

for i in range(len(pages_md_split)):
    pages_md_split[i].page_content = " ".join(
        pages_md_split[i].page_content.split()
    )


# =====================================================
# Split Each Lecture into Smaller Chunks
# =====================================================

# Create a CharacterTextSplitter and split the text further within each
# lecture while preserving the course and lecture title metadata.

char_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=500,
    chunk_overlap=50
)

pages_char_split = char_splitter.split_documents(
    pages_md_split
)

# print(pages_char_split)

# We get a list of Document objects.
#
# Each Document contains:
#
# - page_content
# - metadata
#
# The metadata stores the course title and lecture title.


# =====================================================
# OpenAI Client
# =====================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key
)


# =====================================================
# Initialize the Embedding Model
# =====================================================

embedding = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)


# =====================================================
# Embed Individual Text Chunks
# =====================================================

# To demonstrate how embeddings work, we'll apply the embed_query() method
# to three individual strings.
#
# The first string is the page content of the fourth Document in
# pages_char_split.
#
# This chunk is part of the Analysis vs Analytics lecture.
#
# Next, we'll embed the sixth Document in the list, which is also part of
# the same lecture.
#
# Since these chunks come from the same lecture and are positioned relatively
# close together in the text, we expect them to have a strong semantic
# relationship.
#
# The third string is the page content of the nineteenth Document.
#
# This chunk belongs to the second lecture in the DOCX file.
#
# We expect this chunk to be less closely related to the previous two.


# =====================================================
# Create the Vector Representations
# =====================================================

# Use OpenAI's embedding model to create vector representations of the
# selected chunks.

vector_one = embedding.embed_query(
    pages_char_split[3].page_content
)

vector_two = embedding.embed_query(
    pages_char_split[5].page_content
)

vector_three = embedding.embed_query(
    pages_char_split[18].page_content
)


# =====================================================
# Inspect the Embeddings
# =====================================================

# print(vector_one)

# We obtain a list of floating-point numbers that forms the vector
# representation of the fourth Document in the list.
#
# The embedding model we're using produces vectors with a dimensionality
# of 1536.
#
# Because all of the vectors have the same length, we can calculate the
# dot product between each pair to compare their similarity.

# print(len(vector_one))
# 1536

# print(len(vector_two))
# 1536

# print(len(vector_three))
# 1536


# =====================================================
# Compare Vectors from the Same Lecture
# =====================================================

# These two vectors come from chunks in the same lecture.

np.dot(
    vector_one,
    vector_two
)

# print(np.dot(vector_one, vector_two))

# 0.879128449794393


# =====================================================
# Compare Vectors from Different Lectures
# =====================================================

# The following comparisons involve vectors from different lectures.

np.dot(
    vector_one,
    vector_three
)

# print(np.dot(vector_one, vector_three))

# 0.8000235828747088


np.dot(
    vector_two,
    vector_three
)

# print(np.dot(vector_two, vector_three))

# 0.7934993700101879


# =====================================================
# Compare the Similarity Scores
# =====================================================

# We obtain two results that are approximately 0.80.
#
# These values are smaller than 0.88.
#
# This shows that vector_one and vector_two are more closely related to each
# other than either one is to vector_three.


# =====================================================
# Check the Vector Magnitudes
# =====================================================

# Let's confirm that the vectors created by OpenAI's embedding function have
# a magnitude close to 1.
#
# We can calculate this using NumPy's linear algebra module.
#
# The norm() method calculates the magnitude of a vector.

# print(np.linalg.norm(vector_one))

# 0.9999999518969226


# print(np.linalg.norm(vector_two))

# 0.9999999432048746


# print(np.linalg.norm(vector_three))

# 0.9999999688261209


# =====================================================
# Summary
# =====================================================

# In this lesson, we embedded three of the document chunks.
#
# Each text chunk was converted into a numerical vector.
#
# We then used the dot product to compare the semantic similarity between
# different vectors.
#
# The two chunks from the same lecture produced a higher similarity score
# than the chunks from different lectures.
#
# We also confirmed that the generated vectors have magnitudes close to 1.
#
# In this example, we embedded only three of the total document chunks.