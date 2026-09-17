from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
from openai import OpenAI

import os


# =====================================================
# Overview
# =====================================================

# So far, we've introduced two retrieval methods:
#
# - Similarity Search
# - Maximal Marginal Relevance (MMR)
#
# These methods did a good job of retrieving information relevant to a
# user's query, but they aren't expected to work perfectly in every use case.
#
# In this lesson, we'll create a runnable retriever object so that it can be
# plugged into a LangChain Expression Language chain.


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

embedding = OpenAIEmbeddings()


# =====================================================
# Load the Chroma Vector Store
# =====================================================

# Load the documents and embeddings from the locally stored vector database.

vectorstore = Chroma(
    persist_directory="./intro-to-ds-lectures",
    embedding_function=embedding
)

# print(len(vectorstore.get()["documents"]))

# 20


# =====================================================
# Create a Vector Store Retriever
# =====================================================

# One of the available parameters for as_retriever() is search_type.
#
# search_type determines which retrieval method should be used.
#
# Available options include:
#
# - similarity
# - mmr
# - similarity_score_threshold
#
# similarity_score_threshold performs a similarity search while also
# requiring results to meet a minimum relevance threshold.
#
# Another optional parameter is search_kwargs.
#
# search_kwargs allows us to provide settings specific to the selected
# retrieval method, such as:
#
# - k
# - lambda_mult
# - filter
#
# Since we're using MMR search, we'll specify:
#
# - k = 3
# - lambda_mult = 0.7

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.7
    }
)


# =====================================================
# Inspect the Retriever
# =====================================================

# print(retriever)

# This creates a VectorStoreRetriever from the Chroma vector store using
# MMR search.
#
# If we trace the class hierarchy, we'll find that VectorStoreRetriever
# inherits from the Runnable class.
#
# Because it is a Runnable, it can be used inside LangChain Expression
# Language chains.

# Example output:
#
# vectorstore=Chroma(
#     tags=['Chroma', 'OpenAIEmbeddings'],
#     vectorstore=<langchain_community.vectorstores.chroma.Chroma object ...>
# )
# search_type='mmr'
# search_kwargs={
#     'k': 3,
#     'lambda_mult': 0.7
# }


# =====================================================
# Define the Question
# =====================================================

question = "What software do data scientists use?"


# =====================================================
# Invoke the Retriever
# =====================================================

# All Runnable objects implement the invoke() method.
#
# Here we pass the question directly into the retriever.

retrieved_documents = retriever.invoke(question)

# print(retrieved_documents)


# =====================================================
# Inspect the Retrieved Documents
# =====================================================

# retrieved_documents is a list of Document objects.
#
# We can use a for loop to display the page content and lecture title
# for each retrieved chunk.

for document in retrieved_documents:
    print(
        f"Page Content: {document.page_content}\n"
        f"---------------\n"
        f"Lecture Title: {document.metadata['Lecture Title']}\n"
    )


# =====================================================
# Retrieved Results
# =====================================================

# The invoke() method returns the same chunks we retrieved when using the
# MMR search method in the previous lesson.

# Page Content:
#
# As you can see from the infographic, R, and Python are the two most popular
# tools across all columns. Their biggest advantage is that they can manipulate
# data and are integrated within multiple data and data science software
# platforms. They are not just suitable for mathematical and statistical
# computations. In other words, R, and Python are adaptable. They can solve
# a wide variety of business and data-related problems from beginning to the
# end.
#
# ---------------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# It’s actually a software framework which was designed to address the
# complexity of big data and its computational intensity. Most notably,
# Hadoop distributes the computational tasks on multiple computers, which
# is basically the way to handle big data nowadays. Power BI, SAS, Qlik,
# and especially Tableau are top-notch examples of software designed for
# business intelligence visualizations.
#
# ---------------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Great! We hope we gave you a good idea about the level of applicability
# of the most frequently used programming and software tools in the field
# of data science. Thank you for watching!
#
# ---------------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# =====================================================
# Summary
# =====================================================

# In this lesson, we converted our Chroma vector store into a runnable
# retriever.
#
# We:
#
# - Loaded the existing Chroma vector store.
# - Created a VectorStoreRetriever using as_retriever().
# - Configured the retriever to use MMR search.
# - Set k to 3.
# - Set lambda_mult to 0.7.
# - Invoked the retriever directly with a user's question.
# - Received a list of relevant Document objects.
#
# Since VectorStoreRetriever is a Runnable, we can now use it directly
# as part of a LangChain Expression Language chain.
#
# We now have all of the components necessary to create our first retrieval
# chain and move into the final stage of RAG: generation.