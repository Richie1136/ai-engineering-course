from langgraph.graph import START, END, StateGraph, add_messages
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from collections.abc import Sequence
from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal, Annotated

import os


# =====================================================
# Overview
# =====================================================

# We are gradually moving toward the main goal of this course:
# equipping a chatbot with memory.
#
# In the previous section, we outlined the essential components of a graph
# and the steps required to create one.
#
# These steps included:
#
# - Defining the graph schema using a TypedDict
# - Adding nodes
# - Adding edges
# - Creating routing functions for conditional edges
# - Compiling the graph
# - Visualizing the graph
# - Invoking the graph
#
# Everything worked until we noticed an important limitation:
#
# Our graph did not retain messages between nodes.
#
# By default, when the graph updates a value in the state, the previous
# value is replaced by the new value.
#
# In this lesson, we'll begin discussing graph memory by changing this
# behavior.
#
# Instead of replacing previous messages, we'll append new messages to the
# existing list.
#
# We'll accomplish this using:
#
# - add_messages
# - Annotated


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
# Get Familiar with add_messages
# =====================================================

# We'll continue using our chatbot graph containing:
#
# - Three regular nodes
# - A start node
# - An end node
# - Three regular edges
# - One conditional edge


# =====================================================
# Define the Graph State
# =====================================================

# The graph state stores the data that travels between nodes.
#
# Here we'll store a sequence of LangChain message objects.
#
# We now want to override the default behavior where a new state value
# replaces the existing value.
#
# Instead, we want new messages to be appended to the existing messages.
#
# We accomplish this by modifying the type hint for the messages field.
#
# The messages field is no longer only:
#
# Sequence[BaseMessage]
#
# Instead, we wrap it with Annotated.
#
# Annotated allows us to attach additional metadata to a type.
#
# For example, imagine defining an age variable as an integer.
#
# Although an integer can technically contain any whole number, it wouldn't
# make sense for a person's age to be negative or greater than 120.
#
# Annotated could attach additional information describing this constraint.
#
# The value is still an integer, but it carries extra information that a
# framework or validation tool could use.
#
# In our case, the annotation is the add_messages function.
#
# The messages field still stores a sequence of BaseMessage objects.
#
# However, whenever LangGraph updates this field, it will use add_messages
# to combine the existing messages with the new messages instead of simply
# replacing them.
#
# add_messages is an example of a reducer function.
#
# A reducer defines how existing state should be combined with new incoming
# data.

class State(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]


# =====================================================
# Create the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365
)


# =====================================================
# Basic add_messages Example
# =====================================================

# add_messages() accepts two parameters.
#
# The first parameter contains the existing messages.
#
# The second parameter contains the new messages that should be added.
#
# Here, the first list contains:
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

# print(my_list)

# We receive a list containing three LangChain messages, as expected.

# Example:
#
# [
#     HumanMessage(
#         content="Hi! I'm Robert.",
#         ...
#     ),
#
#     AIMessage(
#         content='Hey, Robert. How can I assist you?',
#         ...
#     ),
#
#     HumanMessage(
#         content="Could you summarize today's news?",
#         ...
#     )
# ]


# =====================================================
# Create the ask_question Node
# =====================================================

# We'll now modify the ask_question node to verify that the graph state is
# updated as expected.
#
# The function first prints all messages currently stored in the state.
#
# It then asks the user for a new question.
#
# The returned state contains:
#
# - An AIMessage containing the chatbot's question
# - A HumanMessage containing the user's response
#
# Because messages uses add_messages as its reducer, these new messages will
# be appended to the existing state rather than replacing it.

def ask_question(state: State) -> State:

    print("\n--------------------> ENTERING ask_question")

    # Display all messages currently stored in the state.

    for message in state["messages"]:
        message.pretty_print()

    question = "What is your question?"

    print(question)

    return State(
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

# The chatbot node also prints all messages currently stored in the state.
#
# The complete message history is then passed to the chat model.

def chatbot(state: State) -> State:

    print("\n---------------> Entering Chatbot")

    for message in state["messages"]:
        message.pretty_print()

    response = chat.invoke(
        state["messages"]
    )

    response.pretty_print()

    return State(
        messages=[
            response
        ]
    )


# =====================================================
# Create the ask_another_question Node
# =====================================================

# This node prints the complete conversation so far.
#
# It then asks whether the user wants to ask another question.
#
# The returned state contains:
#
# - An AIMessage containing the question
# - A HumanMessage containing the user's response

def ask_another_question(state: State) -> State:

    print(
        "\n------------> Entering ask_another_question"
    )

    for message in state["messages"]:
        message.pretty_print()

    question = (
        "Would you like to ask one more question (yes/no)?"
    )

    print(question)

    return State(
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
# Define the Routing Function
# =====================================================

# Previously, each node replaced the messages stored in the state.
#
# Because we are now appending messages, the user's most recent response is
# stored at the end of the list.
#
# Therefore, the routing function must check the last message instead of
# the first one.

def routing_function(
    state: State
) -> Literal["ask_question", "__end__"]:

    if state["messages"][-1].content == "yes":
        return "ask_question"

    return "__end__"


# =====================================================
# Create the Graph
# =====================================================

graph = StateGraph(State)


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
#                 *
#                 *
#                 *
#           +---------+
#           | chatbot |
#           +---------+
#                 *
#                 *
#                 *
#      +----------------------+
#      | ask_another_question |
#      +----------------------+
#             *          .
#             *           .
#             *            .
#      +--------------+   +---------+
#      | ask_question |   | __end__ |
#      +--------------+   +---------+


# =====================================================
# Test the Graph
# =====================================================

graph_compiled.invoke(
    State(
        messages=[]
    )
)


# =====================================================
# Example Execution
# =====================================================

# --------------------> ENTERING ask_question
#
# What is your question?
#
# Could you give me a book by William Shakespeare?


# ---------------> Entering Chatbot
#
# ================================== Ai Message ==================================
#
# What is your question?
#
# ================================ Human Message =================================
#
# Could you give me a book by William Shakespeare?
#
# ================================== Ai Message ==================================
#
# Shakespeare wrote plays and sonnets rather than "books."
# A widely read example is:
#
# Hamlet
#
# - Type: Tragedy (c. 1600–1602)
#
# - Short description:
#   Prince Hamlet seeks to avenge his father's murder after the ghost of the
#   king reveals the crime. The play explores revenge, madness, mortality,
#   and political corruption.
#
# - Famous lines:
#   "To be, or not to be: that is the question."
#   "Frailty, thy name is woman."
#
# - Main characters:
#   Hamlet, King Claudius, Queen Gertrude, Ophelia, Polonius, and Horatio.
#
# Would you like the full text of Hamlet, a short summary, a scene, or a
# recommendation for a comedy or sonnet instead?


# ------------> Entering ask_another_question
#
# Would you like to ask one more question (yes/no)?


# =====================================================
# How the State Changes
# =====================================================

# When we enter the chatbot node, it receives the AIMessage and HumanMessage
# created by ask_question.
#
# The chatbot generates another AIMessage and appends it to the existing
# message history.
#
# When we enter ask_another_question, the state contains the entire
# conversation up to that point.
#
# The node then appends:
#
# - The question asking whether the user wants to continue
# - The user's response
#
# If the user enters "yes", routing_function checks the final message in the
# list and routes the graph back to ask_question.
#
# The existing conversation remains in the state because add_messages keeps
# appending new messages rather than replacing the previous ones.


# =====================================================
# Summary
# =====================================================

# In this lesson, we introduced reducer functions.
#
# By default, LangGraph replaces a state value when a node returns an update.
#
# We changed this behavior for the messages field using:
#
# Annotated[
#     Sequence[BaseMessage],
#     add_messages
# ]
#
# add_messages acts as a reducer that combines the existing message history
# with new messages.
#
# This allows the graph to preserve the conversation across multiple nodes.
#
# Because the conversation history grows over time, the routing function now
# checks the most recent message using:
#
# state["messages"][-1]
#
# instead of checking the first message.