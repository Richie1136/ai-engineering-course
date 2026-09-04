from dotenv import load_dotenv
import os
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


# =====================================================
# Overview
# =====================================================

# In the previous lesson, we introduced the RunnableParallel class.
#
# RunnableParallel allows us to invoke more than one runnable at the same time
# using the same input.
#
# We also visualized the process with an ASCII graph and compared the execution
# time of invoking runnables in parallel versus sequentially.
#
# In this lesson, we'll extend that discussion by passing the books and projects
# from the parallel chains into a new prompt template that estimates how long
# it would take to complete them.


# =====================================================
# Load the API Key
# =====================================================

# Load the environment variables from the .env file.

load_dotenv()

# Retrieve the OpenAI API key.

api_key = os.getenv("OPENAI_API_KEY")


# =====================================================
# Create the Chat Prompt Templates
# =====================================================

# We'll start with two chains.
#
# The first suggests three intermediate-level programming books.
#
# The second suggests three intermediate-level programming projects.
#
# Both chains accept the same input:
#
# - programming language

chat_template_books = ChatPromptTemplate.from_template("""
Suggest three of the best intermediate-level {programming language} books.
Answer only by listing the books.
""")

chat_template_projects = ChatPromptTemplate.from_template("""
Suggest three interesting {programming language} projects suitable for intermediate-level programmers.
Answer only by listing the projects.
""")


# =====================================================
# Initialize the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365
    # max_completion_tokens=900
)


# =====================================================
# Create the String Output Parser
# =====================================================

string_parser = StrOutputParser()


# =====================================================
# Create the Two Chains
# =====================================================

chain_books = (
    chat_template_books
    | chat
    | string_parser
)

chain_projects = (
    chat_template_projects
    | chat
    | string_parser
)


# =====================================================
# Create a RunnableParallel
# =====================================================

# The books key maps to chain_books.
#
# The projects key maps to chain_projects.

chain_parallel = RunnableParallel({
    "books": chain_books,
    "projects": chain_projects
})


# =====================================================
# Invoke the Parallel Chain
# =====================================================

# Both chains receive the same input value.

chain_parallel.invoke({
    "programming language": "python"
})

# Print the results from both chains.

# print(
#     chain_parallel.invoke({
#         "programming language": "python"
#     })
# )

# The output is a dictionary containing suggested books and projects.

# Example:
#
# {
#     "books": (
#         "Fluent Python — Luciano Ramalho\n"
#         "Effective Python: 90 Specific Ways to Write Better Python — Brett Slatkin\n"
#         "Python Cookbook (3rd Edition) — David Beazley and Brian K. Jones"
#     ),
#
#     "projects": (
#         "- Static site generator in Python\n"
#         "- Personal finance manager and dashboard\n"
#         "- Deployable recommendation service"
#     )
# }


# =====================================================
# Graph the Parallel Process
# =====================================================

# Print the ASCII graph of the parallel chain.

# print(chain_parallel.get_graph().print_ascii())

# Parallel<books,projects>Input represents invoking the RunnableParallel
# object with the value "python".
#
# Both ChatPromptTemplate objects are then invoked in parallel.
#
# Next come the ChatOpenAI models, followed by the StrOutputParser objects.
#
# Finally, we receive the output from both chains.

#             +-------------------------------+
#             | Parallel<books,projects>Input |
#             +-------------------------------+
#                    ***               ***
#                 ***                     ***
#               **                           **
# +--------------------+              +--------------------+
# | ChatPromptTemplate |              | ChatPromptTemplate |
# +--------------------+              +--------------------+
#            *                                   *
#            *                                   *
#            *                                   *
#     +------------+                      +------------+
#     | ChatOpenAI |                      | ChatOpenAI |
#     +------------+                      +------------+
#            *                                   *
#            *                                   *
#            *                                   *
#   +-----------------+                 +-----------------+
#   | StrOutputParser |                 | StrOutputParser |
#   +-----------------+                 +-----------------+
#                    ***               ***
#                       ***         ***
#                          **     **
#             +--------------------------------+
#             | Parallel<books,projects>Output |
#             +--------------------------------+


# =====================================================
# batch() vs RunnableParallel
# =====================================================

# It's worth noting the similarities and differences between the batch()
# method and the RunnableParallel class.
#
# batch() allows us to invoke the same runnable using different input values.
#
# RunnableParallel allows us to execute several different runnables using the
# same input value.


# =====================================================
# Measure Sequential Execution Time
# =====================================================

# First, invoke chain_books.

start_time = time.perf_counter()

chain_books.invoke({
    "programming language": "python"
})

end_time = time.perf_counter()

# print(
#     f"Chain Books: "
#     f"{end_time - start_time:.2f} seconds"
# )

# Chain Books: 4.41 seconds


# Next, invoke chain_projects.

start_time = time.perf_counter()

chain_projects.invoke({
    "programming language": "python"
})

end_time = time.perf_counter()

# print(
#     f"Chain Projects: "
#     f"{end_time - start_time:.2f} seconds"
# )

# Chain Projects: 5.92 seconds


# =====================================================
# Measure Parallel Execution Time
# =====================================================

start_time = time.perf_counter()

chain_parallel.invoke({
    "programming language": "python"
})

# This invoke() call outputs a dictionary with books and projects as the keys.
#
# Their corresponding values are the results of invoking chain_books and
# chain_projects.
#
# Our next task is to feed this dictionary into a new ChatPromptTemplate.

end_time = time.perf_counter()

# print(
#     f"Chain Parallel: "
#     f"{end_time - start_time:.2f} seconds"
# )

# Chain Parallel: 6.44 seconds


# =====================================================
# Compare the Execution Times
# =====================================================

# The two sequential calls took approximately:
#
# 4.41 + 5.92 = 10.33 seconds
#
# The parallel call took approximately:
#
# 6.44 seconds
#
# Therefore, invoking the runnables in parallel is more time-efficient than
# invoking them sequentially.


# =====================================================
# Create the Time Estimate Prompt Template
# =====================================================

# We'll now pass the books and projects from the parallel chains into a new
# prompt template.
#
# Based on the books from the first chain and the projects from the second,
# we want an estimate of how long they would take to complete.

chat_template_time = ChatPromptTemplate.from_template("""
I'm an intermediate-level programmer.

Consider the following literature:
{books}

Also, consider the following projects:
{projects}

Roughly how much time would it take me to complete the literature and projects?
""")


# =====================================================
# Create the Time Estimate Chain
# =====================================================

# The first component of this chain is the RunnableParallel object.
#
# Its output is a dictionary containing:
#
# - books
# - projects
#
# That dictionary is passed into chat_template_time.
#
# Next come the ChatOpenAI model and the string output parser.

chain_time1 = (
    RunnableParallel({
        "books": chain_books,
        "projects": chain_projects
    })
    | chat_template_time
    | chat
    | string_parser
)


# =====================================================
# Invoke the Time Estimate Chain
# =====================================================

# Print the response generated by chain_time1.

# print(
#     chain_time1.invoke({
#         "programming language": "python"
#     })
# )

# The model's response estimates the amount of time necessary to complete all
# three Python books and projects.


# =====================================================
# Graph the Complete Chain
# =====================================================

# Print the ASCII graph for chain_time1.

# print(chain_time1.get_graph().print_ascii())

# First, RunnableParallel is executed with the input "python".
#
# chain_books and chain_projects are then executed in parallel.
#
# Their outputs are fed into chat_template_time.
#
# Next come the model, parser, and final string output.

#             +-------------------------------+
#             | Parallel<books,projects>Input |
#             +-------------------------------+
#                    ***               ***
#                 ***                     ***
#               **                           **
# +--------------------+              +--------------------+
# | ChatPromptTemplate |              | ChatPromptTemplate |
# +--------------------+              +--------------------+
#            *                                   *
#            *                                   *
#            *                                   *
#     +------------+                      +------------+
#     | ChatOpenAI |                      | ChatOpenAI |
#     +------------+                      +------------+
#            *                                   *
#            *                                   *
#            *                                   *
#   +-----------------+                 +-----------------+
#   | StrOutputParser |                 | StrOutputParser |
#   +-----------------+                 +-----------------+
#                    ***               ***
#                       ***         ***
#                          **     **
#             +--------------------------------+
#             | Parallel<books,projects>Output |
#             +--------------------------------+
#                              *
#                              *
#                              *
#                    +--------------------+
#                    | ChatPromptTemplate |
#                    +--------------------+
#                              *
#                              *
#                              *
#                       +------------+
#                       | ChatOpenAI |
#                       +------------+
#                              *
#                              *
#                              *
#                     +-----------------+
#                     | StrOutputParser |
#                     +-----------------+
#                              *
#                              *
#                              *
#                  +-----------------------+
#                  | StrOutputParserOutput |
#                  +-----------------------+


# =====================================================
# Simplify RunnableParallel Syntax
# =====================================================

# When we pipe a dictionary of runnables into another runnable, LangChain
# automatically converts that dictionary into a RunnableParallel object.
#
# Because of this, we don't always need to explicitly create
# RunnableParallel().

chain_time2 = (
    {
        "books": chain_books,
        "projects": chain_projects
    }
    | chat_template_time
    | chat
    | string_parser
)


# =====================================================
# Invoke the Simplified Chain
# =====================================================

# Print the response generated by chain_time2.

# print(
#     chain_time2.invoke({
#         "programming language": "python"
#     })
# )


# =====================================================
# Key Takeaway
# =====================================================

# When we pipe a dictionary containing multiple runnables into another
# runnable, LangChain automatically wraps that dictionary in RunnableParallel.
#
# This means these two implementations are equivalent:
#
# RunnableParallel({
#     "books": chain_books,
#     "projects": chain_projects
# })
#
# and
#
# {
#     "books": chain_books,
#     "projects": chain_projects
# }
#
# The chain_time2 implementation is the version you'll encounter most often
# because it dramatically simplifies the code.