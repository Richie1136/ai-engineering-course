from langgraph.graph import START, END, StateGraph, add_messages, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage
)

from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal

import os


# =====================================================
# Overview
# =====================================================

# Throughout this course, we've defined our graph state as a TypedDict with
# a key named messages that stores a sequence of chat messages.
#
# We've also used a reducer function that appends each new message to the
# existing list.
#
# However, a graph's state doesn't have to be defined this way.
#
# The key doesn't have to be called messages.
#
# Its value doesn't have to be a sequence of chat messages.
#
# A reducer function, if present, also doesn't necessarily have to append
# new messages.
#
# State can be customized depending on the application's needs.
#
# However, one of the most common configurations for AI-powered graphs is:
#
# - A key named messages
# - A sequence of chat messages
# - A reducer that appends new messages
#
# Because this pattern is so common, LangGraph provides a pre-built state
# class called MessagesState.


# =====================================================
# MessagesState
# =====================================================

# Previously, we may have defined a custom state like this:
#
# class State(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#
# MessagesState already provides this structure, so we no longer need to
# define the custom State class ourselves.


# =====================================================
# Load the API Key
# =====================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# =====================================================
# Create the OpenAI Client
# =====================================================

client = OpenAI(
    api_key=api_key
)


# =====================================================
# Create the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365
)


# =====================================================
# Get Familiar with add_messages()
# =====================================================

# add_messages() accepts two arguments:
#
# - The existing messages
# - The new messages to add
#
# In this example, the first list contains:
#
# - One HumanMessage
# - One AIMessage
#
# The second list contains:
#
# - One HumanMessage

my_list = add_messages(
    [
        HumanMessage(
            "Hi! I'm Robert."
        ),
        AIMessage(
            "Hey, Robert. How can I assist you?"
        )
    ],
    [
        HumanMessage(
            "Could you summarize today's news?"
        )
    ]
)


# =====================================================
# Create the ask_question Node
# =====================================================

# The ask_question node receives a MessagesState object and returns an
# updated MessagesState.
#
# The function:
#
# - Displays all messages currently stored in the state.
# - Asks the user for a new question.
# - Returns an AIMessage containing the prompt.
# - Returns a HumanMessage containing the user's response.
#
# Because MessagesState uses add_messages as its reducer, these messages are
# appended to the existing state rather than replacing it.

def ask_question(state: MessagesState) -> MessagesState:

    # print("\n--------------------> ENTERING ask_question")

    for message in state["messages"]:
        message.pretty_print()

    question = "What is your question?"

    # print(question)

    return MessagesState(
        messages=[
            AIMessage(
                question
            ),
            HumanMessage(
                input()
            )
        ]
    )


# =====================================================
# Create the Chatbot Node
# =====================================================

# The chatbot node receives the full message history.
#
# It passes that history to the chat model and appends the resulting
# AIMessage to the state.

def chatbot(state: MessagesState) -> MessagesState:

    # print("\n---------------> ENTERING chatbot")

    for message in state["messages"]:
        message.pretty_print()

    response = chat.invoke(
        state["messages"]
    )

    response.pretty_print()

    return MessagesState(
        messages=[
            response
        ]
    )


# =====================================================
# Create the ask_another_question Node
# =====================================================

# This node displays the full conversation and asks whether the user wants
# to ask another question.
#
# It returns:
#
# - An AIMessage containing the question
# - A HumanMessage containing the user's response

def ask_another_question(
    state: MessagesState
) -> MessagesState:

    # print(
    #     "\n------------> ENTERING ask_another_question"
    # )

    for message in state["messages"]:
        message.pretty_print()

    question = (
        "Would you like to ask one more question (yes/no)?"
    )

    # print(question)

    return MessagesState(
        messages=[
            AIMessage(
                question
            ),
            HumanMessage(
                input()
            )
        ]
    )


# =====================================================
# Create the Routing Function
# =====================================================

# The routing function receives the current MessagesState.
#
# It checks the content of the most recent message.
#
# If the user answered "yes", the graph returns to ask_question.
#
# Otherwise, the graph ends.

def routing_function(
    state: MessagesState
) -> Literal["ask_question", "__end__"]:

    if state["messages"][-1].content == "yes":
        return "ask_question"

    return "__end__"


# =====================================================
# Define the Graph
# =====================================================

graph = StateGraph(
    MessagesState
)


# =====================================================
# Add the Nodes
# =====================================================

graph.add_node(
    "ask_question",
    ask_question
)

graph.add_node(
    "chatbot",
    chatbot
)

graph.add_node(
    "ask_another_question",
    ask_another_question
)


# =====================================================
# Add the Edges
# =====================================================

graph.add_edge(
    START,
    "ask_question"
)

graph.add_edge(
    "ask_question",
    "chatbot"
)

graph.add_edge(
    "chatbot",
    "ask_another_question"
)

graph.add_conditional_edges(
    source="ask_another_question",
    path=routing_function
)


# =====================================================
# Compile the Graph
# =====================================================

graph_compiled = graph.compile()


# =====================================================
# Display the Graph
# =====================================================

# print(
#     graph_compiled.get_graph().draw_ascii()
# )

#           +-----------+
#           | __start__ |
#           +-----------+
#                  *
#                  *
#                  *
#          +--------------+
#          | ask_question |
#          +--------------+
#           ***         ..
#          *              ..
#        **                 ..
# +---------+                 .
# | chatbot |               ..
# +---------+             ..
#           ***         ..
#              *      ..
#               **   .
#      +----------------------+
#      | ask_another_question |
#      +----------------------+
#                  .
#                  .
#                  .
#             +---------+
#             | __end__ |
#             +---------+


# =====================================================
# Test the Graph
# =====================================================

graph_compiled.invoke(
    MessagesState(
        messages=[]
    )
)


# =====================================================
# Example Conversation
# =====================================================

# --------------------> ENTERING ask_question
#
# What is your question?
#
# Could you give me a grook by Piet Hein?


# ---------------> ENTERING chatbot
#
# ================================== Ai Message ==================================
#
# What is your question?
#
# ================================ Human Message =================================
#
# Could you give me a grook by Piet Hein?
#
# ================================== Ai Message ==================================
#
# Sorry — I can’t provide a full grook by Piet Hein because his poems are
# still under copyright.
#
# I can, however, do one of the following:
#
# - Give a brief summary of a specific grook.
# - Provide a short excerpt.
# - Write an original grook in a similar style.


# ------------> ENTERING ask_another_question
#
# ================================== Ai Message ==================================
#
# Would you like to ask one more question (yes/no)?
#
# ================================ Human Message =================================
#
# yes


# ================================== Ai Message ==================================
#
# What is your question?
#
# ================================ Human Message =================================
#
# Where was the poet born?
#
# ================================== Ai Message ==================================
#
# Piet Hein was born in Copenhagen, Denmark, on December 16, 1905.
#
# Would you like to ask one more question (yes/no)?


# =====================================================
# Message Accumulation
# =====================================================

# Looking at the output, we can see that messages are appended to the state
# at each step as intended.
#
# MessagesState can also be customized further.
#
# We can create a custom class that inherits from MessagesState and add
# additional keys.
#
# Those additional keys can either use reducers or behave normally.


# =====================================================
# Limitation of Keeping the Full Conversation
# =====================================================

# Appending messages to the state allows us to simulate chatbot memory.
#
# However, this approach is only practical for relatively short interactions.
#
# Long-running conversations become increasingly:
#
# - Time consuming
# - Token consuming
#
# This happens because more conversation history is sent to the chatbot at
# every iteration.
#
# One alternative is to trim the message history and keep only the most
# recent messages.


# =====================================================
# Trimming Messages
# =====================================================

# Instead of sending only the last message or the entire conversation
# history, we can keep a limited number of recent messages.
#
# For example, we can keep only the final five messages:
#
# - "What is your question?"
# - The user's question
# - The chatbot's answer
# - "Would you like to ask one more question?"
# - "Yes"


# =====================================================
# Get Familiar with RemoveMessage
# =====================================================

# RemoveMessage is a special message used to delete another message.
#
# To remove a message, we need its ID.
#
# add_messages() automatically assigns IDs to messages, which allows
# RemoveMessage to identify which messages should be removed.

my_list = add_messages(
    [
        AIMessage(
            "What is your question?"
        ),
        HumanMessage(
            "Could you tell me a grook by Piet Hein?"
        ),
        AIMessage(
            "Certainly! Here's a well-known grook by Piet Hein..."
        ),
        AIMessage(
            "Would you like to ask one more question?"
        ),
        HumanMessage(
            "yes"
        ),
        AIMessage(
            "What is your question?"
        ),
        HumanMessage(
            "Where was the poet born?"
        ),
        AIMessage(
            (
                "Piet Hein was born in Copenhagen, Denmark, "
                "on December 16, 1905."
            )
        ),
        AIMessage(
            "Would you like to ask one more question?"
        )
    ],
    [
        HumanMessage(
            "yes"
        )
    ]
)


# =====================================================
# Create RemoveMessage Objects
# =====================================================

# We want to remove every message except the final five.
#
# my_list[:-5] selects all messages except the final five.
#
# For each of those messages, create a RemoveMessage using the original
# message's ID.

remove_messages = [
    RemoveMessage(
        id=message.id
    )
    for message in my_list[:-5]
]


# =====================================================
# Inspect the RemoveMessage Objects
# =====================================================

# print(remove_messages)

# Example:
#
# [
#     RemoveMessage(
#         content='',
#         additional_kwargs={},
#         response_metadata={},
#         id='027dfeee-5c74-4fc6-8017-b9e205fd6413'
#     ),
#     ...
# ]

# We now have a list of RemoveMessage objects.
#
# Each RemoveMessage stores the ID of the original message that should be
# deleted.


# =====================================================
# Remove the Old Messages
# =====================================================

# Pass the existing messages and the RemoveMessage objects into
# add_messages().
#
# The reducer removes the matching messages and leaves the final five.

# print(
#     add_messages(
#         my_list,
#         remove_messages
#     )
# )


# =====================================================
# Remaining Messages
# =====================================================

# After the older messages are removed, the remaining messages are:
#
# - AIMessage("What is your question?")
# - HumanMessage("Where was the poet born?")
# - AIMessage(
#       "Piet Hein was born in Copenhagen, Denmark, on December 16, 1905."
#   )
# - AIMessage("Would you like to ask one more question?")
# - HumanMessage("yes")


# =====================================================
# Summary
# =====================================================

# In this lesson, we learned how MessagesState simplifies one of the most
# common LangGraph state configurations.
#
# MessagesState:
#
# - Contains a messages key.
# - Stores chat messages.
# - Uses add_messages() as its reducer.
#
# We also learned that add_messages() can:
#
# - Append new messages.
# - Preserve existing message history.
# - Work with RemoveMessage objects to delete messages by ID.
#
# Keeping the entire conversation history can become expensive and slow in
# long-running conversations.
#
# RemoveMessage gives us a way to trim older messages while preserving the
# most recent context.