import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, find_dotenv


# =====================================================
# 1. Introduction
# =====================================================

# In the previous lesson, we preprocessed our course dataset
# to prepare it for semantic search.
#
# Our goal is to improve the 365 Data Science platform's
# search engine by retrieving contextually similar courses
# rather than relying exclusively on exact keyword matches.
#
# Now that our data preprocessing is complete, we're ready
# to connect to Pinecone.
#
# In this lesson, we'll:
#
# 1. Load and prepare our course dataset.
# 2. Configure environment variables.
# 3. Initialize the Pinecone client.
# 4. Define our index configuration.
# 5. Create and connect to a Pinecone index.
#
# Before generating embeddings, we'll also explore
# different embedding algorithms in a separate lesson.


# =====================================================
# 2. Loading the Course Dataset
# =====================================================

# Load the course_descriptions.csv file using pandas.
#
# We specify encoding="cp1252" because our CSV file
# uses Windows-1252 encoding.
#
# By default, pandas.read_csv() uses UTF-8 encoding.

files = pd.read_csv(
    "course_descriptions.csv",
    encoding="cp1252"
)


# =====================================================
# 3. Creating Course Descriptions
# =====================================================

# We'll reuse the create_course_description() function
# from our previous lesson.
#
# This function combines information from four columns:
#
# - course_name
# - course_slug
# - course_technology
# - course_topic
#
# Combining these values into one string allows our
# embedding model to consider the course information
# together when generating a numerical representation.


def create_course_description(row):
    return f"""The course name is {row["course_name"]}, the slug is {row["course_slug"]},
the technology is {row["course_technology"]} and the course topic is {row["course_topic"]}
"""


# =====================================================
# 4. Preparing the DataFrame
# =====================================================

# Configure pandas to display up to 106 rows.
#
# This allows us to examine every course in our dataset
# without pandas hiding rows in the middle.

pd.set_option("display.max_rows", 106)

# Apply our function to every row.
#
# axis=1 tells pandas to apply the function row by row.
#
# The generated text will be stored in a new column
# called new_course_description.

files["new_course_description"] = files.apply(
    create_course_description,
    axis=1
)

# Display the generated course descriptions.
# print(files["new_course_description"])


# =====================================================
# 5. Examining the Course Descriptions
# =====================================================

# Example output:
#
# 0      The course name is Introduction to Tableau, th...
# 1      The course name is The Complete Data Visualiza...
# 2      The course name is Introduction to R Programmi...
# 3      The course name is Data Preprocessing with Num...
# 4      The course name is Introduction to Data and Da...
# 5      The course name is Data Cleaning and Preproces...
# 6      The course name is Introduction to Business An...
# 7      The course name is Data Analysis with Excel Pi...
# 8      The course name is SQL, the slug is sql,\nthe ...
# 9      The course name is Credit Risk Modeling in Pyt...
# 10     The course name is Python Programmer Bootcamp,...
# 11     The course name is SQL + Tableau + Python, the...
# 12     The course name is Introduction to Jupyter, th...
# 13     The course name is Statistics, the slug is sta...
# 14     The course name is Mathematics, the slug is ma...
# 15     The course name is Introduction to Excel, the ...
# 16     The course name is Probability, the slug is pr...
# 17     The course name is Starting a Career in Data S...
# 18     The course name is SQL + Tableau, the slug is ...
# 19     The course name is Time Series Analysis with P...
# 20     The course name is Power BI, the slug is intro...
# 21     The course name is Product Management for AI &...
# 22     The course name is Git and GitHub, the slug is...
# 23     The course name is Deep Learning with TensorFl...
# 24     The course name is Deep Learning with TensorFl...
# 25     The course name is Customer Analytics in Pytho...
# 26     The course name is Web Scraping and API Fundam...
# 27     The course name is Introduction to Python, the...
# 28     The course name is Python for Finance, the slu...
# 29     The course name is Machine Learning in Python,...
# 30     The course name is Advanced Microsoft Excel, t...
# 31     The course name is Convolutional Neural Networ...
# 32     The course name is Data Strategy, the slug is ...
# 33     The course name is Fashion Analytics with Tabl...
# 34     The course name is Dates and Times in Python, ...
# 35     The course name is SQL for Data Science Interv...
# 36     The course name is Data Literacy, the slug is ...
# 37     The course name is AI Applications for Busines...
# 38     The course name is Linear Algebra and Feature ...
# 39     The course name is Machine Learning with Naive...
# 40     The course name is Machine Learning in Excel, ...
# 41     The course name is Machine Learning with Suppo...
# 42     The course name is A/B Testing in Python, the ...
# 43     The course name is Machine Learning with Decis...
# 44     The course name is Machine Learning with K-Nea...
# 45     The course name is Machine Learning with Ridge...
# 46     The course name is Data-Driven Business Growth...
# 47     The course name is Corporate Strategy, the slu...
# 48     The course name is Marketing Strategy, the slu...
# 49     The course name is Negotiation, the slug is ne...
# 50     The course name is Management, the slug is man...
# 51     The course name is Persuasion and Influence, t...
# 52     The course name is Excel for Project Managemen...
# 53     The course name is Agile and Scrum Project Man...
# 54     The course name is Project Management, the slu...
# 55     The course name is Blockchain for Business, th...
# 56     The course name is How to Think Like a Data Sc...
# 57     The course name is Power Query and Data Modeli...
# 58     The course name is Introduction to DAX, the sl...
# 59     The course name is Building Business Reports U...
# 60     The course name is The Machine Learning Proces...
# 61     The course name is The Machine Learning Proces...
# 62     The course name is Working with Text Files in ...
# 63     The course name is Intermediate Python Program...
# 64     The course name is Advanced SQL, the slug is a...
# 65     The course name is FP&A: Building a Company's ...
# 66     The course name is Fixed Income Investments, t...
# 67     The course name is Accounting and Financial St...
# 68     The course name is Fundamentals of Financial R...
# 69     The course name is Building a 3-Statement Mode...
# 70     The course name is Financial Ratio Analysis , ...
# 71     The course name is Corporate Finance , the slu...
# 72     The course name is Fundamentals of Financial M...
# 73     The course name is Introduction to Industry an...
# 74     The course name is Fundamentals of Equity Valu...
# 75     The course name is Introduction to Equity Secu...
# 76     The course name is Alternative Investments, th...
# 77     The course name is Advanced Financial Reportin...
# 78     The course name is Advanced Financial Reportin...
# 79     The course name is Macroeconomics, the slug is...
# 80     The course name is Who Does What in Finance, t...
# 81     The course name is Investment Banking, the slu...
# 82     The course name is Discounted Cash Flow Valuat...
# 83     The course name is Multiples Valuation, the sl...
# 84     The course name is LBO Modeling in Excel, the ...
# 85     The course name is Math for Finance, the slug ...
# 86     The course name is Derivatives, the slug is de...
# 87     The course name is Portfolio Management, the s...
# 88     The course name is Technical Analysis, the slu...
# 89     The course name is The Machine Learning Algori...
# 90     The course name is Customer Churn Analysis wit...
# 91     The course name is Machine Learning Deep Dive:...
# 92     The course name is Customer Engagement Analysi...
# 93     The course name is Advanced SQL for Data Engin...
# 94     The course name is Communication and Presentat...
# 95     The course name is Intro to PowerPoint, the sl...
# 96     The course name is Intro to ChatGPT and Genera...
# 97     The course name is Data Analysis in Power BI w...
# 98     The course name is Mastering Key Performance I...
# 99     The course name is Sign-Up Flow Optimization A...
# 100    The course name is Python for Social Media Ana...
# 101    The course name is Intro to NLP for AI, the sl...
# 102    The course name is Data Analysis with ChatGPT,...
# 103    The course name is ChatGPT for Data Science, t...
# 104    The course name is Intro to LLMs, the slug is ...
# 105    The course name is Growth Analysis with SQL, P...


# =====================================================
# 6. Understanding Environment Variables
# =====================================================

# Before connecting to Pinecone, we need to load
# our Pinecone API key.
#
# API keys are sensitive credentials that allow
# applications to authenticate with external services.
#
# We should never hardcode API keys directly
# into our Python scripts.
#
# Instead, we'll store our API key in a .env file.
#
# Example:
#
# PINECONE_API_KEY=your_api_key_here
#
# The python-dotenv package allows us to load
# these values into our Python environment.
#
# IMPORTANT:
#
# Add .env to your .gitignore file to prevent
# accidentally committing credentials to GitHub.


# =====================================================
# 7. Loading Environment Variables
# =====================================================

# find_dotenv() searches for our .env file.
#
# load_dotenv() loads the variables from that file.
#
# override=True allows values from the .env file
# to replace existing environment variables
# with the same names.
#
# This does not automatically identify which
# values are newer. It simply gives the
# .env file priority.

load_dotenv(find_dotenv(), override=True)


# =====================================================
# 8. Initializing the Pinecone Client
# =====================================================

# Now that our environment variables are loaded,
# we can initialize the Pinecone client.
#
# We'll retrieve the API key using os.environ.get().
#
# This allows us to access our API key without
# exposing it directly in the source code.
#
# The Pinecone client allows us to:
#
# - List existing indexes.
# - Create new indexes.
# - Delete indexes.
# - Connect to indexes.
# - Manage vector data.

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)


# =====================================================
# 9. Configuring Our Pinecone Index
# =====================================================

# Before creating our index, we need to define:
#
# 1. Index name
# 2. Vector dimensions
# 3. Similarity metric
#
# We'll use the following configuration:

index_name = "my-index"
dimension = 384
metric = "cosine"

# index_name:
#
# The name of the Pinecone index.
#
# dimension:
#
# The number of numerical values in each vector.
#
# We're using 384 dimensions because our intended
# embedding model produces 384-dimensional vectors.
#
# The index dimensions must match the dimensions
# generated by the embedding model.
#
# metric:
#
# The similarity metric used to compare vectors.
#
# Cosine similarity measures how closely two
# vectors align in direction.
#
# It is commonly used for semantic search.


# =====================================================
# 10. Checking for an Existing Index
# =====================================================

# Before creating our index, we'll check whether
# an index with the same name already exists.
#
# pc.list_indexes() retrieves the indexes
# associated with our Pinecone account.
#
# We can use a list comprehension to extract
# the name of each index.
#
# For example:
#
# [index.name for index in pc.list_indexes()]
#
# If an index with our chosen name already exists,
# we'll delete it before creating a new one.
#
# WARNING:
#
# Deleting an index permanently removes its
# stored vectors and associated data.
#
# This is appropriate only when we intentionally
# want to reset the index.
#
# In our earlier lessons, "my-index" was created
# with 3 dimensions.
#
# We're now recreating it with 384 dimensions
# for our semantic search case study.

if index_name in [index.name for index in pc.list_indexes()]:

    pc.delete_index(index_name)

    print(f"{index_name} successfully deleted")

else:

    print(f"{index_name} not in index list.")


# =====================================================
# 11. Creating the Pinecone Index
# =====================================================

# Now we'll create our Pinecone index.
#
# We'll use pc.create_index() with:
#
# name:
#     The name of the index.
#
# dimension:
#     The number of dimensions in each vector.
#
# metric:
#     The similarity metric used for comparisons.
#
# spec:
#     The cloud provider and region where
#     the index will be hosted.
#
# We'll create a serverless index using:
#
# Cloud: AWS
# Region: us-east-1

pc.create_index(
    name=index_name,
    dimension=dimension,
    metric=metric,
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


# =====================================================
# 12. Connecting to the Index
# =====================================================

# After creating the index, we'll connect to it.
#
# pc.Index() returns an index object that allows
# us to interact with our vector database.
#
# We can use this object to:
#
# - Upsert vectors.
# - Query vectors.
# - Retrieve stored vectors.
# - Delete vectors.
# - Perform similarity searches.

index = pc.Index(index_name)

# IMPORTANT:
#
# A newly created index may take some time
# to become ready.
#
# Before uploading vectors, we should verify
# that the index is ready to accept operations.


# =====================================================
# 13. Preparing for Embedding Generation
# =====================================================

# Our dataset is now preprocessed, and we've
# configured our Pinecone index.
#
# However, we haven't generated any embeddings yet.
#
# Before implementing that step, we'll study
# different embedding algorithms and their
# strengths and weaknesses.
#
# Once we've selected an embedding model,
# we'll convert our course descriptions
# into numerical vectors.
#
# We can then upload those vectors to Pinecone
# and perform semantic searches.


# =====================================================
# 14. Summary
# =====================================================

# In this lesson, we learned how to:
#
# 1. Load and preprocess course descriptions.
# 2. Combine course information into a single text field.
# 3. Store API keys securely using environment variables.
# 4. Load environment variables with python-dotenv.
# 5. Initialize the Pinecone client.
# 6. Configure an index's name, dimensions, and metric.
# 7. Check whether an index already exists.
# 8. Delete and recreate an index when necessary.
# 9. Create a serverless Pinecone index.
# 10. Connect to the index for future vector operations.