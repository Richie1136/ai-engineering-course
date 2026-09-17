from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv
from openai import OpenAI

import os


# =====================================================
# Overview
# =====================================================

# In the previous lesson, we learned how to access and manipulate documents
# inside a vector store, marking the end of our discussion on indexing.
#
# The upcoming lessons focus on the next stages of the RAG process:
#
# - Retrieval
# - Generation
#
# In this lesson, we'll begin the retrieval stage by performing a
# similarity search.


# =====================================================
# Load the API Key
# =====================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# =====================================================
# Create the OpenAI Client
# =====================================================

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
# Load the Chroma Vector Store
# =====================================================

vectorstore = Chroma(
    persist_directory="./intro-to-ds-lectures",
    embedding_function=embedding
)


# =====================================================
# Create a New Document
# =====================================================

added_document = Document(
    page_content=(
        "Alright! So… Let’s discuss the not-so-obvious differences between "
        "the terms analysis and analytics. Due to the similarity of the words, "
        "some people believe they share the same meaning, and thus use them "
        "interchangeably. Technically, this isn’t correct. There is, in fact, "
        "a distinct difference between the two. And the reason for one often "
        "being used instead of the other is the lack of a transparent "
        "understanding of both. So, let’s clear this up, shall we? First, "
        "we will star"
    ),
    metadata={
        "Course Title": "Introduction to Data and Data Science",
        "Lecture Title": "Analysis vs Analytics"
    }
)


# =====================================================
# Add the Document to the Vector Store
# =====================================================

# add_documents() adds the Document object to the vector store and returns
# its generated ID.

# print(
#     vectorstore.add_documents([
#         added_document
#     ])
# )

# ['d5c6ee69-b5aa-4b55-acad-dd6d9b748542']


# =====================================================
# Similarity Search Goal
# =====================================================

# Our goal for this lesson is to:
#
# 1. Define a question related to data science.
#
# 2. Use the embedding function from the vector store to create a vector
#    representation of the question.
#
# 3. Retrieve a specified number of documents that are most relevant to
#    the question.


# =====================================================
# Understand the Relevant Documents
# =====================================================

# If we skim through the lectures in the DOCX file, the first lecture
# discusses the differences between Analysis and Analytics.
#
# The second lecture covers various software tools and programming languages
# commonly used by data scientists.
#
# Since our question will be about programming languages, we expect the
# retrieved documents to come from the second lecture.


# =====================================================
# Relevant Programming Languages
# =====================================================

# Several programming languages are mentioned throughout the lecture.
#
# One section explains that R and Python are the two most popular tools
# across the different areas of data science.
#
# Another section discusses some limitations of R and Python and introduces
# SQL as a language better suited for working with relational databases.
#
# Later, MATLAB is discussed as a useful tool for mathematical functions
# and matrix manipulation.
#
# Toward the end of the lecture, the author also mentions:
#
# - Java
# - JavaScript
# - C
# - C++
# - Scala
#
# Now that we know which chunks should be relevant, we can use Chroma's
# similarity_search() method and compare the results with our expectations.
#
# In the next stage of RAG, these retrieved documents can be passed to an
# LLM to generate a response.


# =====================================================
# Define the Question
# =====================================================

question = "What programming languages do data scientists use?"


# =====================================================
# Perform a Similarity Search
# =====================================================

# The first parameter is query, which contains the user's question.
#
# The second parameter is k, which determines how many documents should
# be returned.
#
# By default, k is 4.
#
# Here we'll set k to 5 so that we can identify a limitation of basic
# similarity search.

retrieved_documents = vectorstore.similarity_search(
    query=question,
    k=5
)


# =====================================================
# Inspect the Retrieved Documents
# =====================================================

# retrieved_documents is a list containing five Document objects.
#
# Use a for loop to display:
#
# - The page content
# - The lecture title

for document in retrieved_documents:
    print(
        f"Page Content: {document.page_content}\n"
        f"--------\n"
        f"Lecture Title: {document.metadata['Lecture Title']}\n"
    )


# =====================================================
# Analyze the Results
# =====================================================

# All five retrieved chunks come from the second lecture, as expected.
#
# The first chunk focuses mainly on Java and Scala.
#
# The second chunk references:
#
# - Java
# - JavaScript
# - C
# - C++
# - Scala
#
# The third chunk emphasizes R and Python as the most popular tools among
# data scientists.
#
# MATLAB also appears in one of the retrieved chunks, although it is not the
# primary focus of that chunk.


# =====================================================
# Limitation of Similarity Search
# =====================================================

# Looking at the retrieved documents, we can identify a potential problem.
#
# Earlier, we duplicated one of the documents when working with the vector
# store.
#
# If both the original document and its duplicate receive the same or very
# similar similarity score, similarity search may retrieve both of them.
#
# The similarity search method isn't sophisticated enough to recognize that
# one of these results is redundant.
#
# This creates a problem because one of our limited retrieval slots could
# have been used for a different chunk containing new and useful information.
#
# Instead, the model may receive repeated information.


# =====================================================
# Example Retrieved Documents
# =====================================================

# Page Content:
#
# What about big data? Apart from R and Python, people working in this area
# are often proficient in other languages like Java or Scala. These two have
# not been developed specifically for doing statistical analyses, however
# they turn out to be very useful when combining data from multiple sources.
# All right! Let’s finish off with machine learning. When it comes to machine
# learning, we often deal with big data.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Thus, we need a lot of computational power, and we can expect people to use
# the languages similar to those in the big data column. Apart from R, Python,
# and MATLAB, other, faster languages are used like Java, JavaScript, C, C++,
# and Scala. Cool. What we said may be wonderful, but that’s not all! By using
# one or more programming languages, people create application software or,
# as they are sometimes called, software solutions, that are adjusted for
# specific business needs.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# As you can see from the infographic, R, and Python are the two most popular
# tools across all columns. Their biggest advantage is that they can manipulate
# data and are integrated within multiple data and data science software
# platforms. They are not just suitable for mathematical and statistical
# computations. In other words, R, and Python are adaptable. They can solve
# a wide variety of business and data-related problems from beginning to end.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Alright! So… How are the techniques used in data, business intelligence,
# or predictive analytics applied in real life? Certainly, with the help of
# computers. You can basically split the relevant tools into two categories—
# programming languages and software. Knowing a programming language enables
# you to devise programs that can execute specific operations. Moreover, you
# can reuse these programs whenever you need to execute the same action.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# More importantly, it will be sufficient for your need to create quick and
# accurate analyses. However, if your theoretical preparation is strong
# enough, you will find yourself restricted by software. Knowing a programming
# language such as R and Python gives you the freedom to create specific,
# ad-hoc tools for each project you are working on.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# =====================================================
# Summary
# =====================================================

# In this lesson, we moved from indexing into the retrieval stage of RAG.
#
# We:
#
# - Loaded an existing Chroma vector store.
# - Defined a user question.
# - Used similarity_search() to retrieve the five most relevant documents.
# - Inspected the content and metadata of the retrieved documents.
# - Confirmed that the retrieved chunks came from the expected lecture.
# - Identified a limitation of similarity search: duplicate or very similar
#   documents can take up multiple retrieval slots.
#
# In the next step, we'll continue exploring retrieval techniques that can
# help address this limitation.