from dotenv import load_dotenv
import os

from langchain_openai.chat_models import ChatOpenAI

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate
)


# =====================================================
# Overview
# =====================================================

# So far we've learned:
#
# • How to use ChatOpenAI
# • System, Human, and AI messages
# • Prompt templates
# • Chat prompt templates
#
# We've also seen that the output of one invoke() method can become
# the input of another.
#
# In this lesson we'll simplify Few-Shot Prompting by using
# LangChain's FewShotChatMessagePromptTemplate class.
#
# Few-shot prompting provides the model with several examples so it
# can imitate the desired style or behavior.
#
# Zero-shot prompting, in contrast, provides no examples.


# =====================================================
# Load the API Key
# =====================================================

load_dotenv()

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
# Create Message Templates
# =====================================================

# Human prompt template

TEMPLATE_HUMAN = (
    "I've recently adopted a {pet}. "
    "Could you suggest some {pet} names?"
)

# AI prompt template

TEMPLATE_AI = "{response}"


# =====================================================
# Create Human and AI Prompt Templates
# =====================================================

message_template_human = HumanMessagePromptTemplate.from_template(
    TEMPLATE_HUMAN
)

message_template_ai = AIMessagePromptTemplate.from_template(
    TEMPLATE_AI
)


# =====================================================
# Create an Example Template
# =====================================================

# Every example consists of:
#
# HumanMessage
# AIMessage

example_template = ChatPromptTemplate.from_messages([
    message_template_human,
    message_template_ai
])


# =====================================================
# Define Few-Shot Examples
# =====================================================

# Each dictionary supplies the values needed by the example template.

examples = [

    {
        "pet": "dog",
        "response":
            "Oh, absolutely. Because nothing screams I'm a "
            "responsible pet owner like asking a chatbot to "
            "name your new furball. How about 'Bark Twain'?"
    },

    {
        "pet": "cat",
        "response":
            "Oh, absolutely. Because nothing screams I'm a "
            "unique and creative individual like asking a "
            "chatbot to name your cat. How about "
            "'Furry McFurFace'?"
    },

    {
        "pet": "fish",
        "response":
            "Oh, absolutely. Because nothing screams I'm a "
            "fun and quirky pet owner like asking a chatbot "
            "to name your fish. How about 'Gill Gates'?"
    }

]


# =====================================================
# Create the Few-Shot Prompt Template
# =====================================================

# This combines:
#
# • The list of examples
# • The example template

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_template
)

# print(few_shot_prompt)


# =====================================================
# Build the Final Chat Prompt
# =====================================================

# The final prompt contains:
#
# 1. Every example
# 2. A new HumanMessage template

chat_template = ChatPromptTemplate.from_messages([
    few_shot_prompt,
    message_template_human
])


# =====================================================
# Fill the Template
# =====================================================

chat_value = chat_template.invoke({
    "pet": "rabbit"
})

# print(chat_value)


# =====================================================
# Display the Generated Messages
# =====================================================

# ChatPromptValue stores a list of chat messages.
#
# Printing them individually is much easier to read.

for message in chat_value.messages:
    print(f"{message.type}: {message.content}\n")


# Example output:
#
# human: I've recently adopted a dog...
#
# ai: Oh, absolutely...
#
# human: I've recently adopted a cat...
#
# ai: Oh, absolutely...
#
# human: I've recently adopted a fish...
#
# ai: Oh, absolutely...
#
# human: I've recently adopted a rabbit...


# =====================================================
# Send the Prompt to the Model
# =====================================================

response = chat.invoke(chat_value)

print(response.content)


# =====================================================
# Summary
# =====================================================

# The workflow is:
#
# 1. Create HumanMessage and AIMessage prompt templates.
#
# 2. Combine them into an example ChatPromptTemplate.
#
# 3. Store several example dictionaries.
#
# 4. Create a FewShotChatMessagePromptTemplate using:
#       • examples
#       • example_prompt
#
# 5. Combine the few-shot prompt with a new HumanMessage template.
#
# 6. Invoke the ChatPromptTemplate to produce a ChatPromptValue.
#
# 7. Pass the ChatPromptValue directly to ChatOpenAI.
#
# Few-shot prompting allows the model to learn the desired response
# style from examples instead of relying entirely on system
# instructions.
#
# This approach is much cleaner than manually creating large numbers
# of HumanMessage and AIMessage objects.
