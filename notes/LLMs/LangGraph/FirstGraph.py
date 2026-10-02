# =====================================================
# Overview
# =====================================================

# In the previous section, we explored Retrieval-Augmented Generation (RAG),
# learning how to:
#
# - Load documents
# - Split them into meaningful chunks
# - Create embeddings
# - Store them inside a vector database
# - Retrieve relevant documents
# - Generate responses using an LLM
#
# We now move on to LangGraph.
#
# LangGraph allows us to build stateful AI applications where information
# flows through a graph of connected nodes.
#
# In this lesson we'll build our first graph consisting of:
#
# - A state
# - One chatbot node
# - A start node
# - An end node
# - Two edges connecting them
#
# Finally, we'll compile and invoke the graph.

# =====================================================
# Imports
# =====================================================

# START and END represent the beginning and end of the graph.
#
# StateGraph is used to construct the graph itself.

from langgraph.graph import START, END, StateGraph

# TypedDict allows us to define the schema of our graph's state.

from typing_extensions import TypedDict

# ChatOpenAI will power our chatbot node.

from langchain_openai.chat_models import ChatOpenAI

# HumanMessage represents user messages.
#
# BaseMessage is used for type hinting.

from langchain_core.messages import HumanMessage, BaseMessage

# We'll verify that the compiled graph is a Runnable.

from langchain_core.runnables import Runnable

# Used for type hinting.

from collections.abc import Sequence

from dotenv import load_dotenv
from openai import OpenAI

from PIL import Image
from io import BytesIO

import os

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
# Define the Graph State
# =====================================================

# The graph's state stores the data that travels between nodes.
#
# Here we'll store a sequence of LangChain message objects.

class State(TypedDict):
    messages: Sequence[BaseMessage]

# Create an initial state containing a single user message.

state = State(
    messages=[
        HumanMessage(
            "Could you tell me the best stock to invest in?"
        )
    ]
)

# print(state)

# Retrieve the first message and display it nicely.

# print(state["messages"][0].pretty_print())

# =====================================================
# Create the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365
)

# Invoking the model directly returns an AIMessage object.

response = chat.invoke(state["messages"])

# print(response)

# To display only the formatted response:

# print(response.pretty_print())

# =====================================================
# Create the Chatbot Node
# =====================================================

# Every node receives the current graph state.
#
# It performs some work.
#
# Then returns the updated state.

def chatbot(state: State) -> State:

    print("\n--------------------> ENTERING chatbot")

    response = chat.invoke(
        state["messages"]
    )

    response.pretty_print()

    return State(
        messages=[response]
    )

# Test the node independently.

# print(chatbot(state))

# =====================================================
# Create the Graph
# =====================================================

# StateGraph uses our State schema.

graph = StateGraph(State)

# Add the chatbot node.
#
# If the first argument were omitted, the function name
# ("chatbot") would automatically become the node name.

graph.add_node(
    "chatbot",
    chatbot
)

# =====================================================
# Add the Edges
# =====================================================

# Connect:
#
# START
#   ↓
# chatbot
#   ↓
# END

graph.add_edge(
    START,
    "chatbot"
)

graph.add_edge(
    "chatbot",
    END
)

# =====================================================
# Compile the Graph
# =====================================================

# Compiling performs validation and converts the graph
# into a runnable object.

graph_compiled = graph.compile()

# Confirm that the compiled graph is runnable.

# print(isinstance(graph_compiled, Runnable))
# True

# =====================================================
# Display the Graph
# =====================================================

# Generate a Mermaid PNG.

# png_data = graph_compiled.get_graph().draw_mermaid_png()

# image = Image.open(BytesIO(png_data))

# image.show()

# =====================================================
# Invoke the Graph
# =====================================================

# Invoke the compiled graph using the initial state.

# print(graph_compiled.invoke(state))

# The returned state now contains the AI response instead
# of the original HumanMessage.

# =====================================================
# Graph Structure
# =====================================================

#           START
#             │
#             ▼
#        +-----------+
#        | chatbot   |
#        +-----------+
#             │
#             ▼
#            END

# =====================================================
# Summary
# =====================================================

# In this lesson we learned how to:
#
# • Define a graph state using TypedDict.
#
# • Create a node as a Python function that receives
#   and returns state.
#
# • Construct a graph using StateGraph.
#
# • Connect nodes using edges.
#
# • Compile the graph into a Runnable object.
#
# • Invoke the compiled graph.
#
# • Visualize the graph using Mermaid.
#
# This is the foundation for everything we'll build in
# LangGraph. Future lessons will introduce conditional
# edges, loops, memory, tools, and more advanced graphs.