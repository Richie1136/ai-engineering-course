import pandas as pd
from sentence_transformers import SentenceTransformer


# =====================================================
# Introduction
# =====================================================

# In this lesson, we'll examine and preprocess the dataset
# for our semantic search case study.
#
# Our primary objective is to improve the 365 Data Science
# platform's search engine.
#
# Instead of relying exclusively on exact keyword matches,
# we want to retrieve results that are contextually similar
# to a user's search query.
#
# Initially, we'll focus on course-level information.
#
# Later, we'll incorporate more granular information
# at the section level to further improve search results.


# =====================================================
# 1. Loading the Dataset
# =====================================================

# We'll begin by loading the course_descriptions.csv file.
#
# The dataset contains information about the courses
# available on the 365 Data Science platform.
#
# We'll use pandas.read_csv() to load the CSV file
# into a DataFrame.
#
# We're specifying encoding="cp1252" because this
# particular CSV file uses Windows-1252 encoding.
#
# By default, pandas.read_csv() uses UTF-8 encoding.

files = pd.read_csv(
    "course_descriptions.csv",
    encoding="cp1252"
)

# Display the dataset.
# print(files)


# =====================================================
# 2. Examining the Dataset
# =====================================================

# Example output:
#
#                                            course_name  ...                           course_description_short
# 0                              Introduction to Tableau  ...  Teaching you how to tell compelling stories wi...
# 1    The Complete Data Visualization Course with Py...  ...  Teaching you how to master the art of creating...
# 2                        Introduction to R Programming  ...  Providing you with the skills to manipulate, a...
# 3                        Data Preprocessing with NumPy  ...  This course will guide you through one of Pyth...
# 4                Introduction to Data and Data Science  ...  Introducing you to the field of data science a...
# ..                                                 ...  ...                                                ...
# 101                                Intro to NLP for AI  ...  Unlock the power of natural language processin...
# 102                         Data Analysis with ChatGPT  ...  Master ChatGPT for data analysis. Boost your p...
# 103                           ChatGPT for Data Science  ...  Learn how to increase your productivity using ...
# 104                                      Intro to LLMs  ...  This LLM course will guide you step-by-step th...
# 105    Growth Analysis with SQL, Python, and Tableau    ...  Master the creation of a growth data dashboard...
#
# [106 rows x 6 columns]
#
# Our dataset contains:
#
# - 106 rows
# - 6 columns
#
# Each row represents a course.
#
# The columns contain information such as:
#
# - Course name
# - Course slug
# - Course technology
# - Course topic
# - Long course description
# - Short course description
#
# This information will help us generate embeddings
# that represent the content of each course.


# =====================================================
# 3. Understanding Our Vectorization Strategy
# =====================================================

# Now that we've loaded the dataset, we need to determine
# how to prepare it for vectorization.
#
# Our dataset contains structured, tabular information.
#
# Traditional relational databases are well suited
# for storing this type of data.
#
# They allow us to organize information into columns,
# filter records, and combine related tables using joins.
#
# However, our objective is to perform semantic search.
#
# This requires converting text into numerical embeddings.


# =====================================================
# 4. Should We Vectorize Each Column Separately?
# =====================================================

# One possible approach would be to generate a separate
# embedding for each text column.
#
# For example, we could generate embeddings for:
#
# - Course name
# - Course slug
# - Course technology
# - Course topic
# - Course descriptions
#
# This is a valid approach.
#
# However, it may not be the most appropriate solution
# for our current course-level search problem.
#
# If we embed each column separately, the information
# about a course becomes distributed across multiple
# vectors.
#
# Instead, we can combine related text fields into
# a single string.
#
# This allows the embedding model to process the
# information together.
#
# For example:
#
# Course Name: Introduction to Python
# Technology: Python
# Topic: Programming
#
# We could combine these values into:
#
# "The course name is Introduction to Python,
#  the technology is Python, and the course
#  topic is Programming."
#
# This provides additional context about the course.
#
# IMPORTANT:
#
# Combining fields doesn't automatically guarantee
# better embeddings or more accurate search results.
#
# The effectiveness of this approach depends on:
#
# - The embedding model
# - The quality of the text
# - The relevance of the combined fields
# - The length of the resulting text
#
# For our case study, combining related fields is
# a reasonable starting point.


# =====================================================
# 5. Creating a Course Description Function
# =====================================================

# We'll create a function called create_course_description().
#
# This function will combine several columns into
# a single piece of text.
#
# The function accepts one DataFrame row as its argument.
#
# It retrieves the following values:
#
# - course_name
# - course_slug
# - course_technology
# - course_topic
#
# We'll use an f-string to construct a readable
# description from these values.
#
# An f-string allows us to insert Python expressions
# directly into a string using curly braces.
#
# For example:
#
# name = "Python"
#
# description = f"The technology is {name}"
#
# Result:
#
# "The technology is Python"
#
# Instead of simply concatenating the column values,
# we'll include descriptive text to make their
# relationships clearer.


def create_course_description(row):
    return f"""The course name is {row["course_name"]}, the slug is {row["course_slug"]},
the technology is {row["course_technology"]} and the course topic is {row["course_topic"]}
"""


# =====================================================
# 6. Creating a New DataFrame Column
# =====================================================

# Now that we've defined our function, we'll apply
# it to every row in the DataFrame.
#
# We'll create a new column called:
#
# new_course_description
#
# This column will contain the combined text
# generated by create_course_description().
#
# We'll use the pandas apply() method.
#
# DataFrame.apply() applies a function along
# a specified axis.
#
# axis=0:
#     Applies the function to each column.
#
# axis=1:
#     Applies the function to each row.
#
# Since our function needs values from multiple
# columns within the same row, we'll use axis=1.

files["new_course_description"] = files.apply(
    create_course_description,
    axis=1
)


# =====================================================
# 7. Examining the New Column
# =====================================================

# Display the newly created descriptions.
# print(files["new_course_description"])

# Example output:
#
# 0       The course name is Introduction to Tableau, t...
# 1       The course name is The Complete Data Visualiz...
# 2       The course name is Introduction to R Programm...
# 3       The course name is Data Preprocessing with Nu...
# 4       The course name is Introduction to Data and D...
#                              ...
# 101     The course name is Intro to NLP for AI, the s...
# 102     The course name is Data Analysis with ChatGPT...
# 103     The course name is ChatGPT for Data Science, ...
# 104     The course name is Intro to LLMs, the slug is...
# 105     The course name is Growth Analysis with SQL, ...
#
# Name: new_course_description, Length: 106, dtype: object


# =====================================================
# 8. Understanding the Preprocessed Data
# =====================================================

# Our DataFrame now contains a new text column
# representing the combined course information.
#
# Before preprocessing:
#
# course_name        = "Introduction to Python"
# course_slug        = "introduction-to-python"
# course_technology  = "Python"
# course_topic       = "Programming"
#
# After preprocessing:
#
# new_course_description =
#
# "The course name is Introduction to Python,
#  the slug is introduction-to-python,
#  the technology is Python and the
#  course topic is Programming"
#
# This combined text can be passed to an
# embedding model.
#
# The embedding model will convert it into
# a numerical vector.
#
# That vector can then be stored in Pinecone
# and used for semantic search.
#
# Note:
#
# We're creating a new text representation
# of each course, not generating embeddings yet.
#
# The actual vectorization will take place
# in a later step.


# =====================================================
# 9. Preparing for Pinecone
# =====================================================

# Our dataset is now ready for the next stage.
#
# We'll use the preprocessed course information
# to generate embeddings.
#
# We can then connect to the Pinecone API
# and upsert the resulting vectors.
#
# Each vector will represent information
# about a particular course.
#
# This will allow us to compare a user's
# search query against the stored course embeddings
# and retrieve semantically similar results.
#
# In the next lesson, we'll connect to Pinecone
# and begin uploading our course data.


# =====================================================
# 10. Summary
# =====================================================

# In this lesson, we learned how to:
#
# 1. Load a CSV file using pandas.read_csv().
#
# 2. Specify the correct file encoding.
#
# 3. Examine structured course information.
#
# 4. Understand different approaches to
#    vectorizing tabular data.
#
# 5. Combine related text columns into
#    a single course description.
#
# 6. Use f-strings to construct readable text.
#
# 7. Apply a function to every DataFrame row
#    using apply(axis=1).
#
# 8. Create a new column containing
#    preprocessed course information.
#
# 9. Prepare the dataset for embedding
#    generation and Pinecone upserting.


# In the previous lesson, we discussed our new dataset,
# which contains descriptions at the course-section level.
#
# When we create our new vector database, we'll generate
# embeddings at the section level rather than the
# course level used previously.

# Load the new course-section dataset using pandas.
#
# We'll use the same Windows-1252 encoding as before.

files = pd.read_csv(
    "course_section_descriptions.csv",
    encoding="cp1252"
)


# Creating Our Course-Section Bridge
#
# Create a unique identifier for each course-section combination.
#
# We'll combine the course ID and section ID with a hyphen.
#
# This gives us a unique identifier while preserving
# the original IDs so we can identify the course
# and section associated with each entry.

files["unique_id"] = (
    files["course_id"].astype(str)
    + "-"
    + files["section_id"].astype(str)
)


# Creating Metadata
#
# We'll add a metadata column to our DataFrame.
#
# When we upload the embeddings to Pinecone,
# this metadata will help us identify the courses
# and sections returned by similarity searches.
#
# This is particularly useful because numerical
# embedding vectors aren't human-readable.
#
# We'll use pandas apply() together with a lambda function.
#
# The lambda function creates a dictionary for each row.
#
# Each dictionary contains:
#
# - course_name
# - section_name
# - section_description
#
# The dictionary keys are strings, while their values
# come from the corresponding DataFrame columns.

files["metadata"] = files.apply(
    lambda row: {
        "course_name": row["course_name"],
        "section_name": row["section_name"],
        "section_description": row["section_description"]
    },
    axis=1
)


# Why Include Section Descriptions in Metadata?
#
# We're including section_description in our metadata
# because we'll need to review the descriptions when
# evaluating the relevance of our search results.
#
# Since we may not be familiar with every course section,
# displaying the descriptions will help us determine
# whether the retrieved results are actually relevant.
#
# This can also benefit students using the search engine.
#
# For example, a recommendation might receive a high
# similarity score but still be unrelated to what
# the student is looking for.
#
# By providing the complete section description,
# students can review the information themselves
# and decide whether the recommended section
# meets their needs.


# Creating Course-Section Embeddings
#
# Next, we'll create a function that combines
# several text columns into a single string.
#
# We'll include:
#
# - Course name
# - Course technology
# - Course description
# - Section name
# - Section description
#
# We're combining the text at the section level.
#
# This means that course-level information will be
# repeated across the embeddings for its sections.
#
# Each section will still have its own unique
# section name and description.


def create_embeddings(row):

    # Use an f-string to combine the course
    # and section information into one text string.

    combined_text = f"""{row['course_name']} {row['course_technology']}
{row['course_description']} {row['section_name']} {row['section_description']}
"""

    # After preprocessing, we'll convert the combined
    # text into a numerical embedding.
    #
    # The model.encode() method generates an embedding
    # representing the semantic content of the text.
    #
    # Setting show_progress_bar=False prevents the
    # progress bar from appearing for each row.

    return model.encode(
        combined_text,
        show_progress_bar=False
    )


# Loading the Embedding Model
#
# We need an embedding model to convert
# our course-section text into numerical vectors.
#
# We'll continue using all-MiniLM-L6-v2,
# which generates 384-dimensional embeddings.

model = SentenceTransformer("all-MiniLM-L6-v2")


# At the end of the case study, we'll try another
# approach and compare the two methods.
#
# The alternative method involves doubling
# the embedding dimensions.
#
# This would require creating another Pinecone index
# with the appropriate dimensions.
#
# Since this introduces additional setup,
# we'll explore that method later.


# Generating Embeddings for Every Course Section
#
# The final column we need is the embedding column.
#
# Once again, we'll use pandas apply() with axis=1
# to process every row in the DataFrame.
#
# For each row, create_embeddings() combines the
# course-section information and generates
# a numerical embedding.
#
# The resulting embeddings will be stored
# in the "embedding" column.

files["embedding"] = files.apply(
    create_embeddings,
    axis=1
)


# In the next lesson, we'll continue working
# with the embeddings and upsert the new
# course-section data into Pinecone.