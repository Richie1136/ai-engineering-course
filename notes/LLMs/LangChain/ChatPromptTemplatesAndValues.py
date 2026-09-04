from dotenv import load_dotenv
import os

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

# WHY: Chat templates keep the structure and roles of a conversation fixed
# while allowing runtime values to change. This prevents repeated string
# assembly and produces typed messages that a chat model can consume directly.
# Mental model: templates + variables -> ChatPromptValue -> chat model.


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
    seed=365,
    max_completion_tokens=1000
)


# =====================================================
# Create String Templates
# =====================================================

# Instead of creating SystemMessage and HumanMessage objects directly,
# we can first create reusable string templates.

TEMPLATE_SYSTEM = "{description}"

TEMPLATE_HUMAN = (
    "I've recently adopted a {pet}. "
    "Could you suggest some {pet} names?"
)


# =====================================================
# Create Message Prompt Templates
# =====================================================

# Create a SystemMessagePromptTemplate from the system string.

message_template_system = SystemMessagePromptTemplate.from_template(
    template=TEMPLATE_SYSTEM
)

# print(type(message_template_system))

# <class 'langchain_core.prompts.chat.SystemMessagePromptTemplate'>

# The prompt attribute stores an instance of PromptTemplate.

# print(message_template_system)


# Create a HumanMessagePromptTemplate from the human string.

message_template_human = HumanMessagePromptTemplate.from_template(
    template=TEMPLATE_HUMAN
)

# print(message_template_human)

# Like the system template, this object contains a PromptTemplate that
# stores:
#
# - The input variables
# - The original string template


# =====================================================
# Create a Chat Prompt Template
# =====================================================

# A ChatPromptTemplate combines multiple message prompt templates into
# a single reusable chat prompt.

chat_template = ChatPromptTemplate.from_messages([
    message_template_system,
    message_template_human
])

# print(chat_template)

# The ChatPromptTemplate stores:
#
# - All input variables from every message template
# - The list of message prompt templates


# =====================================================
# Invoke the Chat Prompt Template
# =====================================================

# invoke() replaces each placeholder with the supplied values.

chat_value = chat_template.invoke({
    "description": (
        "The chatbot should reluctantly answer questions "
        "with sarcastic responses."
    ),
    "pet": "dog"
})

# print(chat_value)

# The result is a ChatPromptValue object.


# =====================================================
# Chat Prompt Value
# =====================================================

# A ChatPromptValue contains fully constructed chat messages.
#
# The placeholders have now been replaced with actual values.
#
# Instead of message prompt templates, the object now contains:
#
# - SystemMessage
# - HumanMessage

# print(chat_value)


# =====================================================
# Pass the Chat Prompt to the Model
# =====================================================

# ChatOpenAI's invoke() method accepts:
#
# - A string
# - A list of chat messages
# - A PromptValue object
#
# Since ChatPromptValue inherits from PromptValue, we can pass it
# directly into the chat model.

response = chat.invoke(chat_value)


# =====================================================
# Display the Response
# =====================================================

# invoke() returns an AIMessage object.

print(response.content)


# =====================================================
# Summary
# =====================================================

# The workflow for chat prompt templates is:
#
# 1. Create string templates.
#
# 2. Convert each string template into a message prompt template.
#
#    - SystemMessagePromptTemplate
#    - HumanMessagePromptTemplate
#
# 3. Combine the message prompt templates into a ChatPromptTemplate.
#
# 4. Call invoke() on the ChatPromptTemplate and provide values for
#    each placeholder.
#
# 5. This produces a ChatPromptValue containing fully constructed
#    SystemMessage and HumanMessage objects.
#
# 6. Pass the ChatPromptValue directly to ChatOpenAI's invoke()
#    method.
#
# 7. The model returns an AIMessage containing the generated response.
#
# The output of one invoke() method becomes the input of another,
# allowing LangChain components to be chained together.
