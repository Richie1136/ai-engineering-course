from openai import OpenAI
import config

# =============================================================================
# OpenAI Client
# =============================================================================

client = OpenAI(api_key=config.api_key)

# =============================================================================
# Limitations of a Language Model's General Knowledge
# =============================================================================

# A language model can answer questions using the knowledge and context
# available to it.
#
# However, it may not have access to private, business-specific, newly created,
# or otherwise unavailable information.
#
# For example, the model may not know which course will be uploaded next to
# the 365 Data Science platform unless that information is provided to it.


def ask_model(prompt):
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        max_output_tokens=256
    )

    return response.output_text.strip()


# =============================================================================
# Example
# =============================================================================

prompt = "What is the next course to be uploaded to 365 Data Science?"

response = ask_model(prompt)

print(response)

# The model may not be able to answer this question accurately because the
# information may be private, unavailable, or not included in the context
# provided to the model.

# =============================================================================
# LangChain
# =============================================================================

# LangChain is an open-source framework that helps developers build
# applications powered by Large Language Models (LLMs).
#
# It allows developers to connect language models with external data sources,
# tools, and other application components.
#
# This is useful when an LLM needs access to information that is not already
# available in its general knowledge or current prompt.

# =============================================================================
# Why Use External Data?
# =============================================================================

# There are many situations where a language model needs access to custom data.
#
# For example, you might want to create:
#
# - A study assistant that can answer questions about a course syllabus.
# - A customer-service chatbot that understands information about a business.
# - An application that can search private documents.
# - A system that answers questions using a specific knowledge base.
#
# By providing the model with relevant external information, it can generate
# answers that are more specific to the application and its users.

# =============================================================================
# Example: Study Assistant
# =============================================================================

# A language model may have broad general knowledge, but you can improve its
# usefulness by giving it access to the complete syllabus or course material
# that you want to study.
#
# The model can then use that information when answering questions and helping
# you review the material.

# =============================================================================
# Example: Business Chatbot
# =============================================================================

# Suppose you want to create a chatbot that assists customers on a business
# website.
#
# The language model may not know:
#
# - What products the business sells.
# - The company's policies.
# - Its prices or services.
# - Business-specific procedures.
# - Recently updated information.
#
# You can provide this information to the application so the model can use it
# when responding to customers.

# =============================================================================
# Loading and Preparing Custom Data
# =============================================================================

# A common retrieval workflow contains the following steps:

# Step 1: Load the Data
#
# Load the documents or other custom information that the application needs.

# Step 2: Split the Data into Chunks
#
# Large documents are divided into smaller chunks because language models have
# limits on how much text they can process at one time.

# Step 3: Create Embeddings
#
# Each text chunk is converted into a numerical embedding.
#
# Embeddings help represent the meaning of the text numerically, making it
# possible to compare chunks and identify which ones are most relevant to a
# user's question.

# Step 4: Store the Embeddings
#
# The embeddings are saved in a vector store.
#
# A vector store can perform similarity searches to find the text chunks that
# are most closely related to a query.

# Step 5: Retrieve Relevant Information
#
# When the user asks a question, the application searches the vector store and
# retrieves the most relevant chunks.

# Step 6: Generate a Response
#
# The retrieved information is provided to the language model as context so it
# can generate an answer based on the custom data.

# =============================================================================
# Additional LangChain Capabilities
# =============================================================================

# LangChain can also help developers build applications that connect language
# models with tools and actions.
#
# For example, an application might:
#
# - Search a knowledge base.
# - Call another API.
# - Retrieve information from a database.
# - Trigger a follow-up action.
#
# LangChain provides model and tool integrations that help developers assemble
# these components into LLM-powered applications.

# =============================================================================
# Course Focus
# =============================================================================

# In this course, we will focus on using LangChain to integrate our own data
# with a Large Language Model.

# =============================================================================
# Summary
# =============================================================================

# LangChain helps developers build applications that combine language models
# with external data and tools.
#
# A common retrieval workflow is:
#
# 1. Load the custom data.
# 2. Split the data into smaller chunks.
# 3. Convert each chunk into an embedding.
# 4. Store the embeddings in a vector store.
# 5. Retrieve the chunks most relevant to the user's question.
# 6. Provide those chunks to the language model as context.
# 7. Generate an answer based on the retrieved information.