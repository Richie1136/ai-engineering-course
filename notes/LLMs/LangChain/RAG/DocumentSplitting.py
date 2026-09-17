from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter


# =====================================================
# Overview
# =====================================================

# Splitting a document into semantically meaningful chunks is essential in
# the RAG process.
#
# It helps reduce the size of the text so that it fits within a model's
# context window limit.
#
# It also breaks the text into chunks focused on specific topics, which can
# improve the quality of the chatbot's responses.
#
# LangChain offers several different algorithms for splitting documents.


# =====================================================
# Character Text Splitting
# =====================================================

# We'll begin with a splitting method based on a predefined number of
# characters.
#
# This means we obtain chunks of roughly equal size.
#
# Before applying this to our document, let's first understand the idea
# conceptually.


# =====================================================
# Character Splitting Example
# =====================================================

# Imagine we're given a document containing 1,500 characters.
#
# We could create a splitter that separates the text into chunks containing
# a maximum of 500 characters each.
#
# This would result in three chunks of approximately equal size.
#
# We've reduced the size of the individual documents from 1,500 characters
# to 500 characters, while increasing the number of documents from one to
# three.


# =====================================================
# Chunk Overlap
# =====================================================

# Another parameter we can specify is chunk overlap.
#
# Chunk overlap represents the number of characters shared between
# consecutive chunks.
#
# Consider our 1,500-character document split into 500-character chunks.
#
# If we introduce an overlap of 50 characters:
#
# - The first chunk remains unchanged.
#
# - The second chunk begins with the last 50 characters from the first chunk
#   and then takes 450 new characters.
#
# - The third chunk begins with the last 50 characters from the second chunk
#   and then takes another 450 new characters.
#
# - If text still remains, the fourth chunk begins with the last 50
#   characters from the third chunk and contains the remaining text.
#
# Introducing chunk overlap increases the number of chunks, but it also
# preserves some context between consecutive chunks.


# =====================================================
# Load the DOCX Document
# =====================================================

# Create an instance of Docx2txtLoader using the document filename.

loader = Docx2txtLoader(
    "Introduction_to_Data_and_Data_Science.docx"
)

# Load the document.

pages = loader.load()


# =====================================================
# Remove Newline Characters
# =====================================================

# Remove unnecessary newline characters from the text.

for i in range(len(pages)):
    pages[i].page_content = " ".join(
        pages[i].page_content.split()
    )


# =====================================================
# Inspect the Document
# =====================================================

# Display the page content of the first and only document in the list.

# print(pages[0].page_content)

# print(len(pages[0].page_content))

# 8259


# =====================================================
# Create the CharacterTextSplitter
# =====================================================

# The separator parameter determines where the text should be split.
#
# By default, the separator is two newline characters.
#
# Since we removed the newline characters from our document, we'll use a
# period as the separator.
#
# chunk_size defines the maximum number of characters in each chunk.
#
# chunk_overlap defines how many characters should overlap between
# consecutive chunks.

char_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=500,
    chunk_overlap=50
)


# =====================================================
# Split the Document
# =====================================================

# split_documents() accepts a list of Document objects, splits them according
# to the configured settings, and returns a new list of Document objects.

pages_char_split = char_splitter.split_documents(pages)

# print(pages_char_split)

# We now have a list containing many more Document objects.

# print(len(pages_char_split))

# 17


# =====================================================
# Calculate the Expected Number of Chunks
# =====================================================

# print(8259 / 500)

# Original text length / chunk size = 16.518

# If we divide the original text length of 8,259 characters by a chunk size
# of 500 characters, we obtain 16 complete chunks and part of a 17th chunk.

# print(0.518 * 500)

# Fractional part * chunk size = 259.0

# Multiplying the fractional part by 500 gives the approximate number of
# characters expected in the final chunk.

# print(len(pages_char_split[-1].page_content))

# 259


# =====================================================
# Effect of Chunk Overlap
# =====================================================

# Introducing an overlap between chunks changes the number of resulting
# splits.
#
# Setting chunk_overlap to 50 results in additional chunks.

# print(len(pages_char_split))

# 19


# =====================================================
# Inspect the First Chunk
# =====================================================

# print(pages_char_split[0].page_content)

# The last sentence may end abruptly, but we expect it to continue in the
# next chunk.

# Analysis vs Analytics Alright! So… Let’s discuss the not-so-obvious
# differences between the terms analysis and analytics. Due to the similarity
# of the words, some people believe they share the same meaning, and thus use
# them interchangeably. Technically, this isn’t correct. There is, in fact,
# a distinct difference between the two. And the reason for one often being
# used instead of the other is the lack of a transparent understanding of
# both. So, let’s clear this up, shall we? First, we will star


# =====================================================
# Inspect the Second Chunk
# =====================================================

# print(pages_char_split[1].page_content)

# The final characters of the first chunk are included at the beginning of
# the next chunk.
#
# For example, the second chunk begins with:
#
# "let’s clear this up, shall we?"
#
# let’s clear this up, shall we? First, we will start with analysis. Consider
# the following…
#
# You have a huge dataset containing data of various types. Instead of
# tackling the entire dataset and running the risk of becoming overwhelmed,
# you separate it into easier-to-digest chunks and study them individually
# and examine how they relate to other parts. And that’s analysis in a
# nutshell.


# =====================================================
# Using a Period as the Separator
# =====================================================

# To avoid chunks ending abruptly, sometimes even in the middle of a word,
# we can use a period as the separator.
#
# A period indicates the end of a sentence.

# print(len(pages_char_split))

# 21

# The chunk length isn't exactly 500 characters.

# print(len(pages_char_split[0].page_content))

# 444

# print(pages_char_split[0].page_content)

# This time, the chunk ends at the end of a sentence instead of ending
# abruptly.
#
# Notice that the final period is missing because separators aren't included
# in the resulting content.

# Analysis vs Analytics Alright! So… Let’s discuss the not-so-obvious
# differences between the terms analysis and analytics. Due to the similarity
# of the words, some people believe they share the same meaning, and thus use
# them interchangeably. Technically, this isn’t correct. There is, in fact,
# a distinct difference between the two. And the reason for one often being
# used instead of the other is the lack of a transparent understanding of
# both.

# As an exercise, you can create a for loop that appends a period to the end
# of each chunk.


# =====================================================
# Markdown Header Text Splitting
# =====================================================

# CharacterTextSplitter isn't the only text splitter LangChain provides.
#
# CharacterTextSplitter divides text based on:
#
# - Separator
# - Maximum chunk size
# - Chunk overlap
#
# It does a good job of reducing document size, but we don't have much control
# over the topics contained in each chunk.
#
# Another approach is to split a document based on its headings.
#
# MarkdownHeaderTextSplitter performs splitting based on Markdown headers.


# =====================================================
# Markdown Headers
# =====================================================

# In Markdown, heading levels are represented using different numbers of
# hash symbols.
#
# The more hash symbols used, the lower the heading level.


# =====================================================
# Load the Markdown-Formatted Document
# =====================================================

loader_doc = Docx2txtLoader(
    "Introduction_to_Data_and_Data_Science_2.docx"
)

pages = loader_doc.load()

# print(pages)

# The result is a list containing a single Document object.


# =====================================================
# Define the Headers
# =====================================================

# headers_to_split_on is a list of tuples.
#
# Each tuple contains two strings:
#
# - The Markdown heading symbol
# - The label we want to assign to that heading
#
# The first tuple uses a single hash symbol and labels it "Course Title".
#
# The second tuple uses two hash symbols and labels it "Lecture Title".

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Course Title"),
        ("##", "Lecture Title")
    ]
)


# =====================================================
# Split the Document by Headers
# =====================================================

# Retrieve the page_content from the first Document object and split the text
# according to the Markdown headings.

pages_markdown_split = md_splitter.split_text(
    pages[0].page_content
)

# print(pages_markdown_split)

# The result is a list containing multiple Document objects.
#
# Each Document contains:
#
# - page_content
# - metadata


# =====================================================
# Markdown Split Metadata
# =====================================================

# The metadata is a dictionary containing two keys:
#
# - Course Title
# - Lecture Title
#
# These are the labels we defined when creating the splitter.
#
# For the first lecture, their respective values are:
#
# Course Title:
# Introduction to Data and Data Science
#
# Lecture Title:
# Analysis vs Analytics
#
# The second Document corresponds to the second lecture from the same course,
# so the value of Lecture Title changes.
#
# The course title and lecture title are no longer included in page_content
# because the document was split based on those headings.