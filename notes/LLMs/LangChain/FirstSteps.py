# =====================================================
# First Steps with the OpenAI API
# =====================================================

# In this section, we'll build our first chatbot using
# Python and OpenAI's API.
#
# We won't be using the LangChain framework just yet.
#
# The goal is to become familiar with the OpenAI API
# because LangChain relies heavily on it.


import os
import openai
from dotenv import load_dotenv


# =====================================================
# Load the API Key
# =====================================================

# Load the environment variables from the .env file.

load_dotenv()

# Retrieve the OpenAI API key.

api_key = os.getenv("OPENAI_API_KEY")


# =====================================================
# Create the OpenAI Client
# =====================================================

# Create an OpenAI client that will communicate
# with OpenAI's servers.

client = openai.OpenAI()


# =====================================================
# Chat Messages
# =====================================================

# A chat completion consists of a list of messages.
#
# Every message contains:
#
# - role
# - content
#
# The role determines who the message belongs to.
#
# Possible roles include:
#
# - system
# - user
# - assistant
# - tool


# =====================================================
# System Messages
# =====================================================

# A system message defines the chatbot's behavior.
#
# It can specify:
#
# - The chatbot's purpose
# - Its personality
# - Output formatting
# - Writing style
# - Any additional instructions
#
# Providing a good system message usually produces
# much better responses.


# Example:
#
# "You will be provided with statements and your task
# is to convert them to standard English."
#
# Or:
#
# "You are Marv, a chatbot that reluctantly answers
# questions with sarcastic responses."


# =====================================================
# User Messages
# =====================================================

# User messages contain the prompts we send to the
# language model.
#
# They can include:
#
# - Questions
# - Instructions
# - Paragraphs
# - Code
# - Documents
# - Translation requests
# - Summarization requests


# =====================================================
# Assistant Messages
# =====================================================

# Assistant messages are previous chatbot responses.
#
# They are commonly used for few-shot prompting,
# where we provide example conversations to teach the
# model the desired behavior.


# Example:
#
# System:
# "Classify the sentiment of each tweet."
#
# User:
# "This new movie is extraordinary!"
#
# Assistant:
# "Positive"
#
# User:
# "This new album is all right."
#
# Assistant:
# "Neutral"
#
# User:
# "This new book could not have been written worse."
#
# Assistant:
# "Negative"
#
# After seeing these examples, the model understands
# the desired output format.


# =====================================================
# Create the Chat Completion
# =====================================================

completion = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "You are Marv, a chatbot that reluctantly "
                "answers questions with sarcastic responses."
            ),
        },
        {
            "role": "user",
            "content": "Could you explain briefly what a black hole is?",
        },
    ],
    max_completion_tokens=1000,
    seed=365,
)

# The returned object is a ChatCompletion object.
#
# It stores:
#
# - The generated response
# - Token usage
# - Model information
# - Finish reason

# =====================================================
# Understanding the ChatCompletion Object
# =====================================================

# The completion variable is an instance of the
# ChatCompletion class.

# The choices parameter stores a list of Choice objects.

# Each Choice object contains the chatbot's response.

# By default, the model returns only one response.

# This behavior can be controlled using the n parameter.

# More responses result in higher token consumption
# and therefore higher cost.

# The response also includes information about:
#
# - Completion tokens
# - Prompt tokens
# - Total tokens


# =====================================================
# Retrieve the Chatbot Response
# =====================================================

# We used a system message to create a chatbot that
# adopts a sarcastic personality.

# We then passed our question as a user message.

# To display the chatbot's response in a more readable
# format, we retrieve:
#
# 1. The choices list
# 2. The first Choice object
# 3. The message
# 4. The content

print(completion.choices[0].message.content)


# =====================================================
# Example Response
# =====================================================

# Response when max_completion_tokens is set to 1000:
#
# Sure — a black hole is basically a region of spacetime
# where gravity is so strong that once you cross a boundary
# called the event horizon, nothing can escape, not even light.
#
# They form when enough mass is concentrated, such as when a
# massive star's core collapses.
#
# General relativity predicts a central singularity, though
# quantum gravity probably changes that picture.
#
# We detect black holes through their effects, including:
#
# - Accretion disks
# - Gravitational lensing
# - Motions of nearby objects
# - Gravitational waves
#
# In short: the universe's ultimate "do not enter" sign.


# =====================================================
# Maximum Completion Tokens
# =====================================================

# One parameter that affects a model's response is
# max_completion_tokens.

# OpenAI sets model prices based on:
#
# - Input tokens
# - Completion tokens

# Input tokens are the tokens we send to the model.

# Completion tokens are the tokens generated by the model.

# Both are capped to prevent users from sending an excessive
# number of tokens and to prevent models from generating
# endless responses.

# It is still useful to have additional control over how
# much text the model generates.

# Models can sometimes produce more information than needed.

# This can become expensive because we pay for the tokens
# generated by the model.

# This pricing applies when using OpenAI models through the API.

# The ChatGPT platform works differently because it is
# subscription-based rather than directly charging by token.


# =====================================================
# Example with a Smaller Token Limit
# =====================================================

# Response when max_completion_tokens is set to 700:
#
# Fine. A black hole is a region of spacetime where gravity
# is so strong that the escape velocity exceeds the speed
# of light.
#
# Anything that crosses the event horizon cannot escape.
#
# They typically form when massive stars collapse, or through
# mergers and direct collapse to form much larger supermassive
# black holes at galaxy centers.
#
# General relativity predicts a central singularity, but
# quantum gravity may change that explanation.
#
# We don't see black holes directly. Instead, we infer their
# existence from their effects on nearby gas, stars, light,
# and gravitational waves.


# =====================================================
# Seed
# =====================================================

# Another parameter that affects determinism is seed.

# This is similar to the seeds we use with traditional
# machine learning algorithms for reproducibility.

# With large language models, identical results are not
# guaranteed.

# However, using the same seed can help make responses
# as similar as possible.


# =====================================================
# Streaming Responses
# =====================================================

# One way to make a chatbot feel more responsive and
# user-friendly is to print the output continuously instead
# of waiting until the entire response has been generated.

# The stream parameter allows us to do this.

# We add stream to the model parameters and set it to True.

# Example:
#
# stream=True

# This causes the response to be returned as it is generated
# instead of waiting for the full response first.


# =====================================================
# Parameters Affecting the Model's Response
# =====================================================

# Some important parameters include:
#
# - Maximum number of completion tokens
# - Level of randomness
# - Seed
# - Streaming
#
# These parameters give us more control over the model's
# behavior and the responses it generates.

# =====================================================
# Understanding the ChatCompletion Object
# =====================================================

# The completion variable is an instance of the
# ChatCompletion class.

# The choices parameter stores a list of Choice objects.

# Each Choice object contains the chatbot's response.

# By default, the model returns only one response.

# This behavior can be controlled using the n parameter.

# More responses result in higher token consumption
# and therefore higher cost.

# The response also includes information about:
#
# - Completion tokens
# - Prompt tokens
# - Total tokens


# =====================================================
# Retrieve the Chatbot Response
# =====================================================

# We used a system message to create a chatbot that
# adopts a sarcastic personality.

# We then passed our question as a user message.

# To display the chatbot's response in a more readable
# format, we retrieve:
#
# 1. The choices list
# 2. The first Choice object
# 3. The message
# 4. The content

print(completion.choices[0].message.content)


# =====================================================
# Example Response
# =====================================================

# Response when max_completion_tokens is set to 1000:
#
# Sure — a black hole is basically a region of spacetime
# where gravity is so strong that once you cross a boundary
# called the event horizon, nothing can escape, not even light.
#
# They form when enough mass is concentrated, such as when a
# massive star's core collapses.
#
# General relativity predicts a central singularity, though
# quantum gravity probably changes that picture.
#
# We detect black holes through their effects, including:
#
# - Accretion disks
# - Gravitational lensing
# - Motions of nearby objects
# - Gravitational waves
#
# In short: the universe's ultimate "do not enter" sign.


# =====================================================
# Maximum Completion Tokens
# =====================================================

# One parameter that affects a model's response is
# max_completion_tokens.

# OpenAI sets model prices based on:
#
# - Input tokens
# - Completion tokens

# Input tokens are the tokens we send to the model.

# Completion tokens are the tokens generated by the model.

# Both are capped to prevent users from sending an excessive
# number of tokens and to prevent models from generating
# endless responses.

# It is still useful to have additional control over how
# much text the model generates.

# Models can sometimes produce more information than needed.

# This can become expensive because we pay for the tokens
# generated by the model.

# This pricing applies when using OpenAI models through the API.

# The ChatGPT platform works differently because it is
# subscription-based rather than directly charging by token.


# =====================================================
# Example with a Smaller Token Limit
# =====================================================

# Response when max_completion_tokens is set to 700:
#
# Fine. A black hole is a region of spacetime where gravity
# is so strong that the escape velocity exceeds the speed
# of light.
#
# Anything that crosses the event horizon cannot escape.
#
# They typically form when massive stars collapse, or through
# mergers and direct collapse to form much larger supermassive
# black holes at galaxy centers.
#
# General relativity predicts a central singularity, but
# quantum gravity may change that explanation.
#
# We don't see black holes directly. Instead, we infer their
# existence from their effects on nearby gas, stars, light,
# and gravitational waves.


# =====================================================
# Seed
# =====================================================

# Another parameter that affects determinism is seed.

# This is similar to the seeds we use with traditional
# machine learning algorithms for reproducibility.

# With large language models, identical results are not
# guaranteed.

# However, using the same seed can help make responses
# as similar as possible.


# =====================================================
# Streaming Responses
# =====================================================

# One way to make a chatbot feel more responsive and
# user-friendly is to print the output continuously instead
# of waiting until the entire response has been generated.

# The stream parameter allows us to do this.

# We add stream to the model parameters and set it to True.

# Example:
#
# stream=True

# This causes the response to be returned as it is generated
# instead of waiting for the full response first.


# =====================================================
# Parameters Affecting the Model's Response
# =====================================================

# Some important parameters include:
#
# - Maximum number of completion tokens
# - Level of randomness
# - Seed
# - Streaming
#
# These parameters give us more control over the model's
# behavior and the responses it generates.