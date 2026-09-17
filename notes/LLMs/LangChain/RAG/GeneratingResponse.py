from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import os


# =====================================================
# Overview
# =====================================================

# We've made significant progress in constructing our retrieval chain.
#
# So far, we've piped a dictionary into the prompt template.
#
# Recall that this produced a StringPromptValue containing the completed
# prompt template.
#
# In this lesson, we'll continue building the chain by adding:
#
# - The chat model
# - The string output parser
#
# This completes the generation stage of our RAG pipeline.


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

# Start by creating a dictionary containing:
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
# The question and context placeholders have been replaced with their
# respective values.


# =====================================================
# Add the Chat Model
# =====================================================

# The next component we pipe into the chain is our ChatOpenAI instance.
#
# The completed prompt becomes the input to the chat model.
#
# The model then generates an AIMessage.

chain3 = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | chat
)


# =====================================================
# Invoke the Chain with the Chat Model
# =====================================================

# print(chain3.invoke(question))

# We obtain an AIMessage object whose content parameter stores the text
# generated by the model.


# =====================================================
# Example AIMessage Response
# =====================================================

# content='Common software and tools (from the provided lecture):
#
# - R and Python — the two most popular tools. They manipulate data,
#   integrate with many data platforms, and are adaptable beyond just
#   mathematical and statistical computations.
#
# - Hadoop — a software framework for big data that distributes
#   computational tasks across multiple computers.
#
# - Business-intelligence / visualization software — examples include
#   Power BI, SaS, Qlik, and especially Tableau.
#
# Resources:
# "Programming Languages & Software Employed in Data Science -
# All the Tools You Need"'

# print("""Common software and tools (from the provided lecture):
#
# - R and Python — the two most popular tools. They manipulate data,
#   integrate with many data platforms, and are adaptable beyond just
#   math/statistics.
#
# - Hadoop — a software framework for big data that distributes
#   computational tasks across multiple computers.
#
# - Business-intelligence / visualization software — examples include
#   Power BI, SaS, Qlik, and especially Tableau.
#
# Resources:
# "Programming Languages & Software Employed in Data Science -
# All the Tools You Need"
# """)

# The model considers the retrieved context and also follows the instruction
# to include the resource.


# =====================================================
# Add the String Output Parser
# =====================================================

# The final component is an instance of StrOutputParser.
#
# This converts the AIMessage returned by the model into a regular Python
# string.

chain4 = (
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

# print(chain4.invoke(question))


# =====================================================
# Example Final Response
# =====================================================

# Common software and tools (from the provided context):
#
# - R and Python — the two most popular tools. They manipulate data, are
#   integrated into many data-science platforms, and are adaptable beyond
#   just mathematical and statistical computations.
#
# - Hadoop — a software framework for big data that distributes computational
#   tasks across multiple computers to handle computational intensity and
#   scale.
#
# - Business-intelligence / visualization tools — Power BI, SaS, Qlik, and
#   especially Tableau are examples designed for BI visualizations.
#
# Resources:
# "Programming Languages & Software Employed in Data Science -
# All the Tools You Need"


# =====================================================
# Stuffing Documents
# =====================================================

# Inserting all retrieved documents directly into the prompt is called
# stuffing.
#
# Stuffing is straightforward and easy to implement.
#
# It can produce excellent model responses in many cases.
#
# However, it isn't suitable when the number of documents or the amount of
# content is so large that it exceeds the model's context window limit.
#
# Another drawback is that research shows models may pay more attention to
# information near the beginning or end of the document list.
#
# As a result, important information in the middle may receive less attention.


# =====================================================
# Document Refinement
# =====================================================

# One alternative to stuffing is document refinement.
#
# This technique addresses context window limitations by passing documents
# to the model one at a time.
#
# After the model generates an answer using the first document, that answer
# is passed back to the model together with the next document and the
# developer's instructions.
#
# The existing answer is then updated using the newly supplied information.
#
# This process continues until all documents have been processed.
#
# The final answer should therefore account for information from all of the
# documents.
#
# Compared with stuffing, however, refinement requires more calls to the LLM,
# which makes it more expensive.


# =====================================================
# RAG Process Recap
# =====================================================

# Throughout this section, we've studied each stage of Retrieval-Augmented
# Generation.
#
# We:
#
# - Loaded a document
# - Split it into smaller chunks
# - Created embeddings
# - Stored those embeddings in a vector database
# - Created a retriever
# - Retrieved documents relevant to a user's question
# - Passed those documents into a prompt
# - Generated a context-aware response
#
# We now have a complete LangChain Expression Language chain that retrieves
# only the documents most relevant to a user's question and uses them to
# generate a context-aware answer.