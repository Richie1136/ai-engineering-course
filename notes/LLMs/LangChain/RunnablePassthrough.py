from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# =====================================================
# Overview
# =====================================================

# Now that we've built a solid foundation in LangChain Expression Language
# syntax and its inner workings, we can move forward by learning how to pipe
# entire chains together.
#
# RunnablePassthrough allows us to pass an input through without changing it.


# =====================================================
# Load the API Key
# =====================================================

# Load the environment variables from the .env file.

load_dotenv()

# Retrieve the OpenAI API key.

api_key = os.getenv("OPENAI_API_KEY")


# =====================================================
# Create the Strategy Prompt Template
# =====================================================

chat_template_strategy = ChatPromptTemplate.from_template("""
Considering the tools provided, develop a strategy for effectively learning
and mastering them:

{tools}
""")


# =====================================================
# Create the Tools Prompt Template
# =====================================================

# We'll create a chain that lists the most essential tools for a given
# profession.
#
# Then we'll feed this output into a second chain whose task will be to
# suggest effective strategies for mastering those tools.
#
# Here, job title is a placeholder for a profession entered by the user.

chat_template_tools = ChatPromptTemplate.from_template("""
What are the five most important tools a {job title} needs?

Answer only by listing the tools.
""")

# print(chat_template_tools)

# By default, from_template() assumes that the template passed as an argument
# is a HumanMessagePromptTemplate.
#
# The job title placeholder becomes an input variable.


# =====================================================
# Initialize the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365,
    max_completion_tokens=900
)


# =====================================================
# Create the String Output Parser
# =====================================================

# The final building block is a string output parser.

string_parser = StrOutputParser()


# =====================================================
# Create the Tools Chain
# =====================================================

# The first component of chain_strategy expects a dictionary where tools is
# the key and the value is the output from the first chain.
#
# Therefore, to connect the two chains, we need to create this dictionary at
# the end of the first chain.
#
# This is where RunnablePassthrough comes into play.
#
# We pipe a dictionary containing a single element onto the end of the first
# chain.
#
# The key is tools, which matches the input variable of
# chat_template_strategy.
#
# The value is an instance of RunnablePassthrough.

chain_tools = (
    chat_template_tools
    | chat
    | string_parser
    | {"tools": RunnablePassthrough()}
)


# =====================================================
# Create the Strategy Chain
# =====================================================

chain_strategy = (
    chat_template_strategy
    | chat
    | string_parser
)


# =====================================================
# Invoke the Tools Chain
# =====================================================

# print(
#     chain_tools.invoke({
#         "job title": "data scientist"
#     })
# )

# Example output:
#
# Python
# R
# SQL
# Git
# Jupyter Notebook


# =====================================================
# Invoke the Strategy Chain
# =====================================================

# print(
#     chain_strategy.invoke({
#         "tools": """
# Python
# R
# SQL
# Git
# Jupyter Notebook
# """
#     })
# )

# As expected, the chat model begins creating a strategy for effectively
# learning the tools provided.


# =====================================================
# Trace the Input and Output of chain_tools
# =====================================================

# Let's step back and trace the input and output of each component in
# chain_tools.
#
# First, we call the chain with a dictionary that maps job title to
# data scientist.
#
# This dictionary is passed to chat_template_tools, which outputs a
# ChatPromptValue object.
#
# The prompt value is then passed into the chat model, which generates an
# AIMessage.
#
# The AIMessage is then passed into the string parser.
#
# The resulting string becomes the input to RunnablePassthrough.
#
# RunnablePassthrough returns the same value without altering it.
#
# The final result is a dictionary that maps tools to the generated string.


# =====================================================
# Combine the Two Chains
# =====================================================

# Invoking chain_strategy with the output of chain_tools connects the two
# runnable sequences.
#
# We can combine them into a single chain.

chain_combined = chain_tools | chain_strategy

# print(
#     chain_combined.invoke({
#         "job title": "data scientist"
#     })
# )


# =====================================================
# LangChain Expression Language Modularity
# =====================================================

# One of the major advantages of LangChain Expression Language is its
# modularity.
#
# We can create smaller chains and then combine them to form longer chains.
#
# Instead of defining chain_tools and chain_strategy separately, we can also
# recreate chain_combined by piping all of the components together directly.


# =====================================================
# Create One Long Chain
# =====================================================

chain_long = (
    chat_template_tools
    | chat
    | string_parser
    | {"tools": RunnablePassthrough()}
    | chat_template_strategy
    | chat
    | string_parser
)

# When chains start becoming uncomfortably long, we can surround the
# expression with parentheses and place each component on a separate line
# for better readability.


# =====================================================
# Visualizing the Chain
# =====================================================

# In the previous lesson, we demonstrated the modularity of LangChain
# Expression Language.
#
# We created two chat prompt templates.
#
# We then defined two chains by piping the respective templates with a chat
# model and a string parser.
#
# Finally, we used RunnablePassthrough to prepare the output of the first
# chain as the input to the next one.
#
# This section focuses on visualizing these chains.


# =====================================================
# Recreate the Individual Chains
# =====================================================

chain_tools = (
    chat_template_tools
    | chat
    | string_parser
    | {"tools": RunnablePassthrough()}
)

chain_strategy = (
    chat_template_strategy
    | chat
    | string_parser
)


# =====================================================
# Print the Chain Graph
# =====================================================

# LangChain allows us to visualize the components inside a chain.
#
# This can be especially helpful when first learning LangChain Expression
# Language.

# print(chain_long.get_graph().print_ascii())

# The chain components are printed in the order in which the data flows
# through them.
#
# This is a straightforward chain to visualize.
#
# Later, we'll see that graphs can branch when working with parallel
# processes.


# Example graph:
#
# +-------------+
# | PromptInput |
# +-------------+
#        *
#        *
#        *
# +--------------------+
# | ChatPromptTemplate |
# +--------------------+
#        *
#        *
#        *
# +------------+
# | ChatOpenAI |
# +------------+
#        *
#        *
#        *
# +-----------------+
# | StrOutputParser |
# +-----------------+
#        *
#        *
#        *
# +-----------------------+
# | StrOutputParserOutput |
# +-----------------------+
#        *
#        *
#        *
# +-------------+
# | Passthrough |
# +-------------+
#        *
#        *
#        *
# +--------------------+
# | ChatPromptTemplate |
# +--------------------+
#        *
#        *
#        *
# +------------+
# | ChatOpenAI |
# +------------+
#        *
#        *
#        *
# +-----------------+
# | StrOutputParser |
# +-----------------+
#        *
#        *
#        *
# +-----------------------+
# | StrOutputParserOutput |
# +-----------------------+