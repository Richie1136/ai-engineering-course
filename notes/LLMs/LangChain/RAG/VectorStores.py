from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv

import os


# =====================================================
# Overview
# =====================================================

# In the previous lessons, we learned how to:
#
# - Load documents
# - Split documents
# - Create embeddings
#
# In the last lesson, we embedded only three of the 20 chunks stored in
# pages_char_split.
#
# In this lesson, we'll embed and store all of the chunks inside a vector
# store.
#
# We'll use Chroma as our vector store.


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
# Inspect the Number of Documents
# =====================================================

# Check the number of Document objects stored in pages_char_split.

# print(len(pages_char_split))

# 20


# =====================================================
# Create the Chroma Vector Store
# =====================================================

# Instead of embedding each document separately, we can use a vector store
# to embed and store all of the documents at once.
#
# LangChain supports integrations with many vector stores.
#
# In this lesson, we'll use Chroma.
#
# We want the vector store to contain:
#
# - All 20 Document objects
# - Their vector representations
#
# documents:
#     The list of Document objects we want to store.
#
# embedding:
#     The embedding model that creates the vector representation of each
#     document.
#
# persist_directory:
#     The local directory where the vector store will be saved.
#
# persist_directory is optional.

vectorstore = Chroma.from_documents(
    documents=pages_char_split,
    embedding=embedding,
    persist_directory="./intro-to-ds-lectures"
)


# =====================================================
# Load an Existing Vector Store
# =====================================================

# When loading an existing vector store, we also provide the same embedding
# model that was used when the vector store was created.
#
# This is necessary because documents can later be updated or new documents
# can be added.
#
# The vector store must use the same embedding function so that existing and
# newly added documents are represented consistently.
#
# If the vector database is used in a context-aware chatbot, the user's query
# must also be embedded using the same embedding function so that the system
# can locate semantically similar documents.

vectorstore_from_directory = Chroma(
    persist_directory="./intro-to-ds-lectures",
    embedding_function=embedding
)

# vectorstore was created using Chroma.from_documents(), which embedded all
# of the Document objects and stored them locally.
#
# vectorstore_from_directory was created by loading the existing Chroma
# database from that local directory.


# =====================================================
# Inspect and Manage Documents
# =====================================================

# In this section, we'll look at how to:
#
# - Retrieve documents
# - Retrieve embeddings
# - Add documents
# - Update documents
# - Delete documents


# =====================================================
# Retrieve Data from the Vector Store
# =====================================================

# print(vectorstore_from_directory.get())

# The get() method returns a dictionary containing information about the
# documents stored in the vector store.


# =====================================================
# Retrieve an Embedding by ID
# =====================================================

# The get() method allows us to specify IDs and choose which information
# should be included.
#
# To retrieve the embedding for a specific document, provide its ID and
# include the embeddings.

# print(
#     vectorstore_from_directory.get(
#         ids="da0d1d62-e20a-4832-af93-e1abdfda28f2",
#         include=["embeddings"]
#     )
# )

# This returns the vector representation associated with the specified ID.


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
#     vectorstore_from_directory.add_documents([
#         added_document
#     ])
# )

# ['ac99504a-ed6a-4ec8-97df-08415e1adcbe']


# =====================================================
# Retrieve the Added Document
# =====================================================

# Use get() with the generated ID to confirm that the document was added.

# print(
#     vectorstore_from_directory.get(
#         "ac99504a-ed6a-4ec8-97df-08415e1adcbe"
#     )
# )

# Example output:
#
# {
#     'ids': ['ac99504a-ed6a-4ec8-97df-08415e1adcbe'],
#     'embeddings': None,
#     'documents': [
#         'Alright! So… Let’s discuss the not-so-obvious differences between '
#         'the terms analysis and analytics...'
#     ],
#     'uris': None,
#     'included': ['metadatas', 'documents'],
#     'data': None,
#     'metadatas': [
#         {
#             'Course Title': 'Introduction to Data and Data Science',
#             'Lecture Title': 'Analysis vs Analytics'
#         }
#     ]
# }


# =====================================================
# Create an Updated Document
# =====================================================

updated_document = Document(
    page_content=(
        "Great! We hope we gave you a good idea about the level of "
        "applicability of the most frequently used programming and software "
        "tools in the field of data science. Thank you for watching!"
    ),
    metadata={
        "Course Title": "Introduction to Data and Data Science",
        "Lecture Title": (
            "Programming languages & Software Employed in Data Science - "
            "All the tools You Need"
        )
    }
)


# =====================================================
# Update the Existing Document
# =====================================================

# update_document() replaces the document stored under the specified ID.
#
# document_id:
#     The ID of the document we want to update.
#
# document:
#     The new Document object that should replace it.

vectorstore_from_directory.update_document(
    document_id="ac99504a-ed6a-4ec8-97df-08415e1adcbe",
    document=updated_document
)


# =====================================================
# Retrieve the Updated Document
# =====================================================

# Confirm that the document was updated correctly.

# print(
#     vectorstore_from_directory.get(
#         "ac99504a-ed6a-4ec8-97df-08415e1adcbe"
#     )
# )

# Example output:
#
# {
#     'ids': ['ac99504a-ed6a-4ec8-97df-08415e1adcbe'],
#     'embeddings': None,
#     'documents': [
#         'Great! We hope we gave you a good idea about the level of '
#         'applicability of the most frequently used programming and '
#         'software tools in the field of data science. Thank you for watching!'
#     ],
#     'uris': None,
#     'included': ['metadatas', 'documents'],
#     'data': None,
#     'metadatas': [
#         {
#             'Course Title': 'Introduction to Data and Data Science',
#             'Lecture Title': (
#                 'Programming languages & Software Employed in Data Science - '
#                 'All the tools You Need'
#             )
#         }
#     ]
# }


# =====================================================
# Delete the Document
# =====================================================

# The final method we'll cover is delete().
#
# delete() removes the document associated with the specified ID.

vectorstore_from_directory.delete(
    "ac99504a-ed6a-4ec8-97df-08415e1adcbe"
)


# =====================================================
# Confirm the Document Was Deleted
# =====================================================

# print(
#     vectorstore_from_directory.get(
#         "ac99504a-ed6a-4ec8-97df-08415e1adcbe"
#     )
# )

# {
#     'ids': [],
#     'embeddings': None,
#     'documents': [],
#     'uris': None,
#     'included': ['metadatas', 'documents'],
#     'data': None,
#     'metadatas': []
# }


# =====================================================
# Summary
# =====================================================

# In this lesson, we:
#
# - Embedded all of our document chunks.
# - Stored the chunks and their embeddings in Chroma.
# - Saved the vector store locally.
# - Reloaded an existing vector store.
# - Retrieved stored information with get().
# - Added a new Document.
# - Updated an existing Document.
# - Deleted a Document.
#
# This completes the vector storage portion of the indexing stage in our
# RAG workflow.