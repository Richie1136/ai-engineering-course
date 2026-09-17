from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
from openai import OpenAI

import os


# =====================================================
# Overview
# =====================================================

# In the previous lesson, we used Chroma's semantic similarity search
# algorithm and found that it performed well when retrieving content
# relevant to a user's prompt.
#
# However, it had an important limitation.
#
# If the vector store contained duplicate documents with a high similarity
# score, similarity search could retrieve both the original and the duplicate.
#
# Retrieving documents based only on similarity can therefore reduce
# diversity and cause the retriever to miss other useful information.
#
# In this lesson, we'll study another retrieval algorithm that addresses
# this problem:
#
# Maximal Marginal Relevance (MMR) Search.


# =====================================================
# Change the Question
# =====================================================

# In the previous lesson, we used the following question:
#
# "What programming languages do data scientists use?"
#
# The similarity search returned chunks mentioning:
#
# - R
# - Python
# - MATLAB
# - Other programming languages commonly used in data science
#
# In this lesson, we'll change the question.

question = "What software do data scientists use?"

# Skimming through the DOCX file, we find software such as:
#
# - Excel
# - SPSS
# - Apache Hadoop
# - Other software tools


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
# Similarity Search
# =====================================================

# Apply similarity_search() to retrieve documents relevant to the query.
#
# Set k=3 so that only three documents are returned.

retrieved_documents = vectorstore.similarity_search(
    query=question,
    k=3
)


# =====================================================
# Inspect the Retrieved Documents
# =====================================================

# retrieved_documents is a list containing three Document objects.
#
# Use a for loop to display:
#
# - The page content
# - The lecture title

for document in retrieved_documents:
    # print(
    #     f"Page Content: {document.page_content}\n"
    #     f"--------\n"
    #     f"Lecture Title: {document.metadata['Lecture Title']}\n"
    # )
    pass


# =====================================================
# Analyze the Similarity Search Results
# =====================================================

# All three chunks come from the second lecture.
#
# However, after inspecting the content:
#
# - The first document focuses on R and Python, which isn't directly
#   relevant to the current question about software.
#
# - The other two documents do not mention the software we expected.
#
# This suggests that similarity search missed important information
# located elsewhere in the text.


# =====================================================
# Example Similarity Search Results
# =====================================================

# Page Content:
#
# As you can see from the infographic, R, and Python are the two most popular
# tools across all columns. Their biggest advantage is that they can manipulate
# data and are integrated within multiple data and data science software
# platforms. They are not just suitable for mathematical and statistical
# computations. In other words, R, and Python are adaptable. They can solve a
# wide variety of business and data-related problems from beginning to the end.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Alright! So… How are the techniques used in data, business intelligence, or
# predictive analytics applied in real life? Certainly, with the help of
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
# What about big data? Apart from R and Python, people working in this area are
# often proficient in other languages like Java or Scala. These two have not
# been developed specifically for doing statistical analyses, however they
# turn out to be very useful when combining data from multiple sources. All
# right! Let’s finish off with machine learning. When it comes to machine
# learning, we often deal with big data.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# =====================================================
# Maximal Marginal Relevance (MMR) Search
# =====================================================

# Maximal Marginal Relevance attempts to balance:
#
# - Relevance to the user's query
# - Diversity among the retrieved documents
#
# A document's relevance is determined by its similarity score with the query.
#
# The higher the similarity score, the more relevant the document is.


# =====================================================
# Redundancy and Diversity
# =====================================================

# A document's redundancy score is calculated by comparing it with all
# documents that have already been retrieved and taking the highest
# similarity score.
#
# A higher redundancy score means the document is less likely to contribute
# new information.
#
# Redundancy:
#
# Max(Similarity())
#
# Diversity:
#
# -Max(Similarity())


# =====================================================
# Marginal Relevance
# =====================================================

# Marginal relevance combines relevance and diversity.
#
# The balance is controlled by lambda, which takes a value between 0 and 1.
#
# Marginal Relevance:
#
# lambda * Similarity()
# - (1 - lambda) * Max(Similarity())
#
# A lambda value closer to 0 places more emphasis on diversity.
#
# A lambda value closer to 1 places more emphasis on relevance.
#
# Therefore, adjusting lambda allows us to decide whether relevance or
# diversity should have more influence on the retrieval results.


# =====================================================
# MMR Search with High Diversity
# =====================================================

# Set lambda_mult to 0.1 to favor document diversity.

retrieved_documents2 = vectorstore.max_marginal_relevance_search(
    query=question,
    k=3,
    lambda_mult=0.1
)

for document in retrieved_documents2:
    # print(
    #     f"Page Content: {document.page_content}\n"
    #     f"--------\n"
    #     f"Lecture Title: {document.metadata['Lecture Title']}\n"
    # )
    pass


# =====================================================
# Analyze the High-Diversity Results
# =====================================================

# We retrieve:
#
# - The familiar document referencing R and Python
# - The closing remarks from the same lecture
# - A paragraph from the Analysis vs Analytics lecture
#
# The final result appears too diverse for our needs.


# =====================================================
# Example High-Diversity Results
# =====================================================

# Page Content:
#
# As you can see from the infographic, R, and Python are the two most popular
# tools across all columns. Their biggest advantage is that they can manipulate
# data and are integrated within multiple data and data science software
# platforms. They are not just suitable for mathematical and statistical
# computations. In other words, R, and Python are adaptable. They can solve a
# wide variety of business and data-related problems from beginning to the end.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Great! We hope we gave you a good idea about the level of applicability of
# the most frequently used programming and software tools in the field of data
# science. Thank you for watching!
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Such as using an analysis to explain how a story ended the way it did or how
# there was a decrease in sales last summer. All this means that we do analyses
# to explain how and/or why something happened. Great! Now, this leads us
# nicely on to the definition of analytics. As you have probably guessed,
# analytics generally refers to the future. Instead of explaining past events
# it explores potential future ones.
#
# --------
#
# Lecture Title:
# Analysis vs Analytics


# =====================================================
# Filter the MMR Search
# =====================================================

# Another available parameter is filter.
#
# We can use it to restrict retrieval to documents from the second lecture.

retrieved_documents3 = vectorstore.max_marginal_relevance_search(
    query=question,
    k=3,
    lambda_mult=0.1,
    filter={
        "Lecture Title": (
            "Programming Languages & Software Employed in Data Science - "
            "All the Tools You Need"
        )
    }
)

for document in retrieved_documents3:
    # print(
    #     f"Page Content: {document.page_content}\n"
    #     f"--------\n"
    #     f"Lecture Title: {document.metadata['Lecture Title']}\n"
    # )
    pass


# =====================================================
# Analyze the Filtered Results
# =====================================================

# Two of the documents appear again.
#
# However, the document from the Analysis vs Analytics lecture has now been
# replaced by a document from the required lecture.
#
# This result is more relevant to the question because it references software
# such as:
#
# - Apache Hadoop
# - Apache HBase
# - MongoDB


# =====================================================
# Example Filtered Results
# =====================================================

# Page Content:
#
# As you can see from the infographic, R, and Python are the two most popular
# tools across all columns. Their biggest advantage is that they can manipulate
# data and are integrated within multiple data and data science software
# platforms. They are not just suitable for mathematical and statistical
# computations. In other words, R, and Python are adaptable. They can solve a
# wide variety of business and data-related problems from beginning to the end.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Among the many applications we have plotted, we can say there is an
# increasing amount of software designed for working with big data such as
# Apache Hadoop, Apache HBase, and MongoDB. In terms of big data, Hadoop is
# the name that must stick with you. Hadoop is listed as software in the sense
# that it is a collection of programs, but don’t imagine it as a nice-looking
# application.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Great! We hope we gave you a good idea about the level of applicability of
# the most frequently used programming and software tools in the field of data
# science. Thank you for watching!
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# =====================================================
# Increase Lambda to 0.7
# =====================================================

# Increasing lambda places more emphasis on relevance and less emphasis on
# diversity.

retrieved_documents4 = vectorstore.max_marginal_relevance_search(
    query=question,
    k=3,
    lambda_mult=0.7,
    filter={
        "Lecture Title": (
            "Programming Languages & Software Employed in Data Science - "
            "All the Tools You Need"
        )
    }
)

for document in retrieved_documents4:
    # print(
    #     f"Page Content: {document.page_content}\n"
    #     f"--------\n"
    #     f"Lecture Title: {document.metadata['Lecture Title']}\n"
    # )
    pass


# =====================================================
# Analyze the 0.7 Lambda Results
# =====================================================

# The first and final documents are retrieved again.
#
# The second document changes to a chunk that references:
#
# - Hadoop
# - Power BI
# - SAS
# - Qlik
# - Tableau


# =====================================================
# Example Lambda 0.7 Results
# =====================================================

# Page Content:
#
# As you can see from the infographic, R, and Python are the two most popular
# tools across all columns. Their biggest advantage is that they can manipulate
# data and are integrated within multiple data and data science software
# platforms. They are not just suitable for mathematical and statistical
# computations. In other words, R, and Python are adaptable. They can solve a
# wide variety of business and data-related problems from beginning to the end.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# It’s actually a software framework which was designed to address the
# complexity of big data and its computational intensity. Most notably, Hadoop
# distributes the computational tasks on multiple computers which is basically
# the way to handle big data nowadays. Power BI, SAS, Qlik, and especially
# Tableau are top-notch examples of software designed for business intelligence
# visualizations.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Great! We hope we gave you a good idea about the level of applicability of
# the most frequently used programming and software tools in the field of data
# science. Thank you for watching!
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# =====================================================
# Set Lambda to 1
# =====================================================

# Setting lambda_mult to 1 removes the diversity component completely.
#
# This means retrieval is based only on relevance.

retrieved_documents5 = vectorstore.max_marginal_relevance_search(
    query=question,
    k=3,
    lambda_mult=1,
    filter={
        "Lecture Title": (
            "Programming Languages & Software Employed in Data Science - "
            "All the Tools You Need"
        )
    }
)

for document in retrieved_documents5:
    print(
        f"Page Content: {document.page_content}\n"
        f"--------\n"
        f"Lecture Title: {document.metadata['Lecture Title']}\n"
    )


# =====================================================
# Analyze the Lambda 1 Results
# =====================================================

# We get a familiar result that closely matches the output from
# similarity_search().
#
# A lambda value of 1 removes the diversity component entirely, leaving
# only relevance.


# =====================================================
# Example Lambda 1 Results
# =====================================================

# Page Content:
#
# As you can see from the infographic, R, and Python are the two most popular
# tools across all columns. Their biggest advantage is that they can manipulate
# data and are integrated within multiple data and data science software
# platforms. They are not just suitable for mathematical and statistical
# computations. In other words, R, and Python are adaptable. They can solve a
# wide variety of business and data-related problems from beginning to the end.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# Page Content:
#
# Alright! So… How are the techniques used in data, business intelligence, or
# predictive analytics applied in real life? Certainly, with the help of
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
# What about big data? Apart from R and Python, people working in this area are
# often proficient in other languages like Java or Scala. These two have not
# been developed specifically for doing statistical analyses, however they
# turn out to be very useful when combining data from multiple sources. All
# right! Let’s finish off with machine learning. When it comes to machine
# learning, we often deal with big data.
#
# --------
#
# Lecture Title:
# Programming Languages & Software Employed in Data Science -
# All the Tools You Need


# =====================================================
# Summary
# =====================================================

# Similarity search retrieves documents based only on relevance to the query.
#
# This can cause problems when highly similar or duplicate documents occupy
# multiple retrieval slots.
#
# Maximal Marginal Relevance attempts to balance:
#
# - Relevance
# - Diversity
#
# lambda_mult controls that balance.
#
# Lower lambda values:
#
# - Favor diversity
# - Can retrieve less relevant documents
#
# Higher lambda values:
#
# - Favor relevance
# - Reduce diversity
#
# Setting lambda_mult to 1 removes the diversity component entirely.
#
# We can also use metadata filters to restrict retrieval to a specific subset
# of documents.