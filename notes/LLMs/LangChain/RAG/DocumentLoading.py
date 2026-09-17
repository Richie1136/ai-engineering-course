from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import copy


# =====================================================
# Overview
# =====================================================

# LangChain offers a large collection of document loaders.
#
# Once you understand the loading process, you can work with virtually any
# supported document format.
#
# In this lesson, we'll demonstrate loading:
#
# - PDF files
# - DOCX files
#
# We'll also perform a small amount of preprocessing on the PDF text.


# =====================================================
# Load a PDF Document
# =====================================================

# Create an instance of the PyPDFLoader class.

loader_pdf = PyPDFLoader("Introduction_to_Data_and_Data_Science.pdf")

# We'll eventually use this PDF to create a chatbot that accepts a student's
# question, retrieves the relevant information from the course transcript,
# and answers the question.

pages_pdf = loader_pdf.load()

# print(pages_pdf)

# The result is a list of Document objects.
#
# Each Document contains:
#
# - page_content
# - metadata


# =====================================================
# Document Structure
# =====================================================

# page_content
#
# Stores the text extracted from that page.
#
# metadata
#
# Stores information about the document such as:
#
# - File path
# - Page number


# =====================================================
# Removing Unnecessary Newlines
# =====================================================

# After inspecting the PDF, we notice that the transcription introduced
# many unnecessary newline characters.
#
# One disadvantage of excessive newlines is increased token consumption,
# which increases API cost.
#
# Since modifying the original list isn't ideal, we'll first create a
# deep copy.

pages_pdf_cut = copy.deepcopy(pages_pdf)

# This allows us to modify the copied Document objects without changing
# the originals stored in pages_pdf.


# =====================================================
# Remove Newlines from a Single Page
# =====================================================

# First retrieve the page_content from one document.
#
# split()
#
# Splits the text into individual words while removing whitespace and
# newline characters.
#
# ' '.join(...)
#
# Joins all words back together using a single space as the separator.

# print(' '.join(pages_pdf_cut[0].page_content.split()))


# =====================================================
# Remove Newlines from Every Document
# =====================================================

# Apply the same preprocessing to every Document object.

for i in pages_pdf_cut:
    i.page_content = " ".join(i.page_content.split())

# print(pages_pdf_cut)


# =====================================================
# Why This Preprocessing?
# =====================================================

# This type of preprocessing isn't always necessary.
#
# Every document requires different preprocessing depending on its
# formatting.
#
# In this example, the newlines clutter the text instead of improving
# readability, so removing them is beneficial.
#
# In many other documents, however, newline characters help preserve
# structure and can later be used to split the text into meaningful chunks.


# =====================================================
# Load a DOCX Document
# =====================================================

# In the previous example we loaded a PDF.
#
# We'll now demonstrate loading a Microsoft Word document.

loader_docx = Docx2txtLoader(
    "Introduction_to_Data_and_Data_Science.docx"
)

pages_docx = loader_docx.load()

# print(pages_docx)

# We receive a list containing a single Document object.
#
# Unlike the PDF loader, the DOCX loader loads the entire document into
# one Document.


# =====================================================
# DOCX Metadata
# =====================================================

# The metadata only contains the document path.
#
# Since a DOCX document doesn't naturally contain page boundaries like a
# PDF, there is no page number stored.
#
# In our particular use case, this isn't a disadvantage because the
# document represents the transcript of a video course.
#
# A student asking questions about the course wouldn't normally reference
# a specific page number.


# =====================================================
# Summary
# =====================================================

# We learned how to load documents using LangChain document loaders.
#
# PyPDFLoader
#
# - Loads PDFs
# - Returns one Document per page
# - Metadata includes the page number
#
# Docx2txtLoader
#
# - Loads Microsoft Word (.docx) files
# - Returns a single Document containing the entire file
# - Metadata contains the file path
#
# We also preprocessed the PDF by removing unnecessary newline characters
# to reduce token usage and create cleaner text for later stages of the
# RAG pipeline.