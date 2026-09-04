from dotenv import load_dotenv
import os

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# Overview
# =====================================================

# By default, chat models such as ChatOpenAI return AIMessage objects.
#
# While AIMessage objects are useful because they contain metadata such as
# token usage and model information, there are many situations where we only
# need the generated text itself.
#
# LangChain provides Output Parsers that convert model outputs into more
# convenient Python data types.
#
# Examples include:
#
# • String Output Parser
# • DateTime Output Parser
# • Comma-Separated List Output Parser
#
# In this lesson we'll use the String Output Parser, which extracts the text
# content from an AIMessage and returns it as a standard Python string.
#
# Why convert it? Downstream Python code often expects a string so it can save
# the answer, display it in a user interface, search it, or pass it to another
# function. The full AIMessage is more useful only when we also need metadata.


# =====================================================
# Load the API Key
# =====================================================

# Load the environment variables from the .env file.

load_dotenv()

# Retrieve the OpenAI API key.

api_key = os.getenv("OPENAI_API_KEY")

# This variable makes it easy to inspect whether the key was found, but we
# should never print the secret itself. ChatOpenAI can also read
# OPENAI_API_KEY directly from the environment, so we do not need to pass the
# variable to the constructor below.


# =====================================================
# Initialize the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    # Low randomness makes a small teaching example easier to reproduce.
    temperature=0,
    seed=365
)


# =====================================================
# Create a Human Message
# =====================================================

message_human = HumanMessage(
    content="Can you give me an interesting fact I probably didn't know about?"
)


# =====================================================
# Invoke the Chat Model
# =====================================================

# The response returned by ChatOpenAI is an AIMessage object.
# We use a list because chat models accept an ordered conversation made of
# system, human, and AI messages. This example contains only one human message.

response = chat.invoke([message_human])

# print(response)


# Example Output
#
# AIMessage(
#     content="Cleopatra lived closer in time to the Moon landing..."
# )


# =====================================================
# Create a String Output Parser
# =====================================================

# The String Output Parser extracts only the content of the AIMessage.
# It gives every chain the same small conversion step instead of making each
# caller know that generated text lives in response.content.

str_output_parser = StrOutputParser()


# =====================================================
# Parse the Response
# =====================================================

response_parsed = str_output_parser.invoke(response)

# At this point, response is still an AIMessage, while response_parsed is a
# normal str. The parser does not rewrite or improve the answer; it only changes
# the representation used by the rest of the program.

# print(response_parsed)


# Example Output
#
# "Octopuses extensively edit their own RNA...
# This allows them to modify how genes are
# expressed without changing their DNA."


# =====================================================
# Summary
# =====================================================

# The workflow is:
#
# 1. Send a prompt to ChatOpenAI.
#
# 2. Receive an AIMessage object.
#
# 3. Create a StrOutputParser.
#
# 4. Pass the AIMessage to the parser.
#
# 5. Receive a standard Python string.
#
# StrOutputParser is useful whenever an application only needs the generated
# text rather than the complete AIMessage object and its associated metadata.
#
# Mental model:
# HumanMessage -> ChatOpenAI -> AIMessage -> StrOutputParser -> str
