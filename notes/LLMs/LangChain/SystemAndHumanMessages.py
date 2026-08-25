from dotenv import load_dotenv
import os

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# =====================================================
# System and Human Messages
# =====================================================

# In this lesson, we'll take a closer look at the SystemMessage and
# HumanMessage classes.
#
# These correspond to the "system" and "user" roles that we previously
# used with OpenAI's Chat Completions API.
#
# Each message object represents who is sending the message to the model.


# =====================================================
# Load the API Key
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
    seed=365
)


# =====================================================
# Create a System Message
# =====================================================

# A SystemMessage provides high-level instructions that guide the
# model's behavior throughout the conversation.
#
# It can define the model's:
# - Personality
# - Tone
# - Purpose
# - Response style
#
# For example, we can instruct the model to respond sarcastically.

message_system = SystemMessage(
    content="You are Marv, a chatbot that reluctantly answers questions with sarcastic responses."
)


# =====================================================
# Create a Human Message
# =====================================================

# A HumanMessage represents the user's prompt.
#
# The message content is provided through the content parameter.

message_human = HumanMessage(
    content="I've recently adopted a dog. Could you suggest some dog names?"
)

# =====================================================
# Invoke the Chat Model
# =====================================================

# Unlike our previous examples where invoke() accepted a single string,
# it can also accept a list of chat messages.
#
# Here we provide:
# 1. The system message
# 2. The human message

response = chat.invoke([
    message_system,
    message_human
])


# =====================================================
# Display the Response
# =====================================================

# invoke() returns an AIMessage object.
#
# The generated text is stored inside the content attribute.

print(response.content)


# =====================================================
# Summary
# =====================================================

# SystemMessage
# - Defines the chatbot's behavior.
# - Usually appears first.
# - Sets the personality, role, or instructions for the model.
#
# HumanMessage
# - Represents the user's request.
# - Contains the prompt the model should answer.
#
# invoke()
# - Can accept a list of chat messages.
# - Returns an AIMessage containing the model's response.
#
# By combining different message types, we can create much more
# controlled and structured conversations with the language model.