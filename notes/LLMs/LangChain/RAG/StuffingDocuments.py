from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from openai import OpenAI

import os


# =====================================================
# Overview
# =====================================================

# This is the final step of the RAG technique:
#
# Generation
#
# We'll create a LangChain Expression Language chain from scratch that uses
# the documents returned by the retriever to generate a context-specific
# response.


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
# Load the Chroma Vector Store
# =====================================================

# Create a vector store object using the locally stored Intro to Data Science
# lectures.

vectorstore = Chroma(
    persist_directory="./intro-to-ds-lectures",
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )
)

# print(len(vectorstore.get()["documents"]))

# 20


# =====================================================
# Create the Retriever
# =====================================================

# Create a vector-store-backed retriever using as_retriever().
#
# We'll use the same settings from the VectorStoreRetriever lesson.
#
# The retriever implements invoke() and can therefore be used as part of a
# LangChain Expression Language chain.

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.7
    }
)


# =====================================================
# Create the Prompt Template
# =====================================================

# Define the prompt template using a string.
#
# The prompt contains:
#
# - The user's question
# - Context returned by the retriever
# - Instructions for displaying the lecture resource

TEMPLATE = """
Answer the following question:
{question}

To answer the question, use only the following context:
{context}

At the end of the response, specify the name of the lecture this context is taken from in the format:
Resources: "Lecture Title"
where "Lecture Title" should be substituted with the title of all resource lectures.
"""


# =====================================================
# Create the PromptTemplate Object
# =====================================================

prompt_template = PromptTemplate.from_template(
    TEMPLATE
)


# =====================================================
# Initialize the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365
)


# =====================================================
# Define the Question
# =====================================================

question = "What software do data scientists use?"


# =====================================================
# Create the First Part of the Chain
# =====================================================

# We now have all the components required to begin constructing our chain.
#
# Start with a dictionary containing:
#
# - context
# - question
#
# The context value is the retriever.
#
# When invoked, the retriever returns a list of relevant Document objects.
#
# RunnablePassthrough returns the user's question without changing it.

chain = RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
})


# =====================================================
# Invoke the RunnableParallel
# =====================================================

# print(chain.invoke(question))

# The result contains:
#
# - context: the three retrieved Document objects
# - question: the original user question

# Example:
#
# {
#     'context': [
#         Document(
#             metadata={
#                 'Course Title': 'Introduction to Data and Data Science',
#                 'Lecture Title':
#                     'Programming Languages & Software Employed in Data Science - '
#                     'All the Tools You Need'
#             },
#             page_content=(
#                 'As you can see from the infographic, R, and Python are the '
#                 'two most popular tools across all columns...'
#             )
#         ),
#
#         Document(
#             metadata={
#                 'Course Title': 'Introduction to Data and Data Science',
#                 'Lecture Title':
#                     'Programming Languages & Software Employed in Data Science - '
#                     'All the Tools You Need'
#             },
#             page_content=(
#                 'It’s actually a software framework which was designed to '
#                 'address the complexity of big data... Power BI, SaS, Qlik, '
#                 'and especially Tableau are top-notch examples...'
#             )
#         ),
#
#         Document(
#             metadata={
#                 'Course Title': 'Introduction to Data and Data Science',
#                 'Lecture Title':
#                     'Programming Languages & Software Employed in Data Science - '
#                     'All the Tools You Need'
#             },
#             page_content=(
#                 'Great! We hope we gave you a good idea about the level of '
#                 'applicability of the most frequently used programming and '
#                 'software tools in the field of data science...'
#             )
#         )
#     ],
#
#     'question': 'What software do data scientists use?'
# }


# =====================================================
# Pipe the Dictionary into the Prompt Template
# =====================================================

# Continue constructing the chain by piping the dictionary into the prompt
# template.
#
# Because the dictionary is now being piped into another runnable, LangChain
# automatically converts it into a RunnableParallel.
#
# This means we no longer need to explicitly wrap the dictionary inside
# RunnableParallel.
#
# This makes the code cleaner.

chain2 = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt_template
)


# =====================================================
# Invoke the Prompt Chain
# =====================================================

# print(chain2.invoke(question))

# The result is a StringPromptValue containing the completed prompt.
#
# The question and context placeholders have now been replaced with their
# respective values.


# =====================================================
# Example Filled-In Prompt
# =====================================================

# Answer the following question:
# What software do data scientists use?
#
# To answer the question, use only the following context:
#
# [
#     Document(
#         metadata={
#             'Lecture Title':
#                 'Programming Languages & Software Employed in Data Science - '
#                 'All the Tools You Need',
#             'Course Title': 'Introduction to Data and Data Science'
#         },
#         page_content=(
#             'As you can see from the infographic, R, and Python are the two '
#             'most popular tools across all columns...'
#         )
#     ),
#
#     Document(
#         metadata={
#             'Lecture Title':
#                 'Programming Languages & Software Employed in Data Science - '
#                 'All the Tools You Need',
#             'Course Title': 'Introduction to Data and Data Science'
#         },
#         page_content=(
#             'It’s actually a software framework which was designed to address '
#             'the complexity of big data... Power BI, SaS, Qlik, and especially '
#             'Tableau are top-notch examples...'
#         )
#     ),
#
#     Document(
#         metadata={
#             'Course Title': 'Introduction to Data and Data Science',
#             'Lecture Title':
#                 'Programming Languages & Software Employed in Data Science - '
#                 'All the Tools You Need'
#         },
#         page_content=(
#             'Great! We hope we gave you a good idea about the level of '
#             'applicability of the most frequently used programming and '
#             'software tools in the field of data science...'
#         )
#     )
# ]
#
# At the end of the response, specify the name of the lecture this context
# is taken from in the format:
#
# Resources: "Lecture Title"
#
# where "Lecture Title" should be substituted with the title of all
# resource lectures.


# =====================================================
# Complete the RAG Chain
# =====================================================

# The next step is to add:
#
# - The chat model
# - The string output parser
#
# The prompt created by prompt_template becomes the input to ChatOpenAI.
#
# ChatOpenAI generates an AIMessage.
#
# StrOutputParser extracts the text from that AIMessage.

chain3 = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | chat
    | StrOutputParser()
)


# =====================================================
# Invoke the Complete RAG Chain
# =====================================================

# print(chain3.invoke(question))


# =====================================================
# Summary
# =====================================================

# In this lesson, we entered the generation stage of RAG.
#
# We:
#
# - Loaded the existing Chroma vector store.
# - Created an MMR retriever.
# - Created a prompt template containing question and context placeholders.
# - Used RunnablePassthrough to preserve the original user question.
# - Used the retriever to populate the context.
# - Combined both values using a RunnableParallel-style dictionary.
# - Passed the completed prompt into ChatOpenAI.
# - Used StrOutputParser to return the final answer as a string.
#
# The retrieved documents are inserted directly into the prompt as context.
#
# This approach is known as stuffing documents into the prompt.