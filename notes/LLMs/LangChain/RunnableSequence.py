from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# =====================================================
# Overview
# =====================================================

# In the previous lesson, we learned how easily chains can be composed using
# the LangChain Expression Language.
#
# Recall that LCEL uses a pipe symbol to connect components, meaning the output
# of one component becomes the input of the next.
#
# We also demonstrated that we could apply the invoke() method to each
# component, allowing for transparency and traceability.
#
# The same idea also applies to the batch() and stream() methods.
#
# This lesson focuses on identifying the types of LangChain objects that can
# be integrated into a chain and understanding the type of object produced
# when those components are chained together.


# =====================================================
# Runnable and RunnableSequence Classes
# =====================================================

# Load the environment variables from the .env file.

load_dotenv()

# Retrieve the OpenAI API key.

api_key = os.getenv("OPENAI_API_KEY")


# =====================================================
# Initialize the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365,
    max_completion_tokens=700
)


# =====================================================
# Create the Chat Prompt Template
# =====================================================

# This template includes two input variables:
#
# - pet
# - breed

chat_template = ChatPromptTemplate.from_messages([
    (
        "human",
        "I've recently adopted a {pet} which is a {breed}. "
        "Could you suggest several training tips?"
    )
])


# =====================================================
# Create the Chain
# =====================================================

# Create the chain by piping the chat template into the chat model.

chain = chat_template | chat


# =====================================================
# Examine the Chain Object
# =====================================================

# chain is an instance of the RunnableSequence class.

print(type(chain))

# <class 'langchain_core.runnables.base.RunnableSequence'>

# RunnableSequence is responsible for implementing a sequence of runnables
# where the output of each runnable becomes the input of the next.


# =====================================================
# Examine the Chat Prompt Template
# =====================================================

print(type(chat_template))

# <class 'langchain_core.prompts.chat.ChatPromptTemplate'>

# ChatPromptTemplate is not directly an instance of the Runnable class.

# It is derived from classes that eventually inherit from Runnable.

# Because of this inheritance, ChatPromptTemplate has access to methods such as:
#
# - invoke()
# - batch()
# - stream()


# =====================================================
# Runnable Objects
# =====================================================

# A Runnable is a unit of work that can be:
#
# - Invoked
# - Batched
# - Streamed
# - Transformed
# - Composed

# Each runnable implements methods such as:
#
# - invoke()
# - batch()
# - stream()
# - transform()

# Runnables can also be composed.

# This means multiple runnable objects can be linked together to form a chain.

# Chains themselves are also runnables.

# This means that chains can also be composed with other runnables to create
# even longer chains.


# =====================================================
# Runnable Inputs and Outputs
# =====================================================

# Recall that invoke() accepts different types of inputs depending on the
# object it is being applied to.

# Each runnable can also return a different type of output.


# =====================================================
# Prompt Components
# =====================================================

# A prompt component accepts a dictionary as input.

# It outputs a PromptValue object.


# =====================================================
# Chat and Language Models
# =====================================================

# A chat model or language model accepts a PromptValue.

# It outputs either:
#
# - A chat message
# - A string


# =====================================================
# Output Parsers
# =====================================================

# An output parser transforms chat messages or strings into other formats.

# These formats can include:
#
# - Strings
# - Lists
# - DateTime objects


# =====================================================
# Other Runnable Components
# =====================================================

# The final two components are:
#
# - Retriever
# - Tool