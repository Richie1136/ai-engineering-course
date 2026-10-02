from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from collections.abc import Sequence
from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal

import os


# =====================================================
# Overview
# =====================================================

# In the previous lessons, we implemented a simple graph containing:
#
# - A start node
# - A chatbot node
# - An end node
# - Two edges connecting them
#
# We'll now make the graph more complex by adding:
#
# - An ask_question node
# - An ask_another_question node
# - A conditional edge
#
# The new flow will work like this:
#
# START
#   ↓
# ask_question
#   ↓
# chatbot
#   ↓
# ask_another_question
#   ↓
# conditional routing
#
# If the user enters "yes", the graph loops back to ask_question.
#
# Otherwise, the graph proceeds to END.


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

# The graph state stores the data that travels between nodes.
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


# =====================================================
# Create the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365
)


# =====================================================
# Create the Chatbot Node
# =====================================================

# Every node receives the current graph state.
#
# It performs some work and returns an updated state.

def chatbot(state: State) -> State:

    print("\n--------------------> ENTERING chatbot")

    response = chat.invoke(
        state["messages"]
    )

    response.pretty_print()

    return State(
        messages=[response]
    )


# =====================================================
# Create the ask_question Node
# =====================================================

# This node prompts the user to ask a question.
#
# It accepts a State object and returns another State object.
#
# The returned state contains a HumanMessage storing the user's input.

def ask_question(state: State) -> State:

    print("\n--------------------> ENTERING ask_question")

    print("What is your question?")

    return State(
        messages=[
            HumanMessage(
                input()
            )
        ]
    )


# =====================================================
# Test ask_question
# =====================================================

# print(
#     ask_question(
#         State(messages=[])
#     )
# )

# Example:
#
# --------------------> ENTERING ask_question
# What is your question?
# Can you tell me how to make money with AI?
#
# {
#     'messages': [
#         HumanMessage(
#             content='Can you tell me how to make money with AI?'
#         )
#     ]
# }


# =====================================================
# Create the ask_another_question Node
# =====================================================

# This node asks whether the user wants to ask another question.
#
# The user's answer is stored inside a HumanMessage and returned as the
# updated state.

def ask_another_question(state: State) -> State:

    print("\n--------------------> ENTERING ask_another_question")

    print("Would you like to ask one more question (yes/no)?")

    return State(
        messages=[
            HumanMessage(
                input()
            )
        ]
    )


# =====================================================
# Test ask_another_question
# =====================================================

# print(
#     ask_another_question(
#         State(messages=[])
#     )
# )

# Example:
#
# --------------------> ENTERING ask_another_question
# Would you like to ask one more question (yes/no)?
# yes
#
# {
#     'messages': [
#         HumanMessage(
#             content='yes'
#         )
#     ]
# }


# =====================================================
# Define the Routing Function
# =====================================================

# We now have all of our nodes.
#
# However, one of the edges will behave differently.
#
# It will be a conditional edge.
#
# The routing function receives the current state and decides which node
# should run next.
#
# Literal allows us to specify the exact string values that the function
# is expected to return.
#
# Here, the function can return:
#
# - "ask_question"
# - "__end__"

def routing_function(
    state: State
) -> Literal["ask_question", "__end__"]:

    # Check the HumanMessage returned by ask_another_question.
    #
    # If the user entered "yes", route back to ask_question.

    if state["messages"][0].content == "yes":
        return "ask_question"

    # Otherwise, terminate the graph.

    return "__end__"


# =====================================================
# Create the Graph
# =====================================================

# We now have all of the components needed to construct the graph:
#
# - ask_question
# - chatbot
# - ask_another_question
# - routing_function

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
# Add the Regular Edges
# =====================================================

# Connect:
#
# START
#   ↓
# ask_question
#   ↓
# chatbot
#   ↓
# ask_another_question

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


# =====================================================
# Add the Conditional Edge
# =====================================================

# The source parameter identifies the node where conditional routing begins.
#
# After ask_another_question finishes, LangGraph evaluates the routing
# function.
#
# The routing function determines which node should run next.

graph.add_conditional_edges(
    source="ask_another_question",
    path=routing_function
)

# A path_map could also be used when the routing function returns values
# that need to be mapped to specific nodes.
#
# Example:
#
# graph.add_conditional_edges(
#     source="ask_another_question",
#     path=routing_function,
#     path_map={
#         "True": "ask_question",
#         "False": "__end__"
#     }
# )


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

# Example:
#
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

# Invoke the compiled graph using an empty list of messages.
#
# The ask_question node will populate the state with the user's first
# question.

print(
    graph_compiled.invoke(
        State(
            messages=[]
        )
    )
)


# =====================================================
# Example Execution
# =====================================================

# --------------------> ENTERING ask_question
# What is your question?
# Could you tell me how to make money with AI?
#
# --------------------> ENTERING chatbot
#
# ================================== Ai Message ==================================
#
# Short answer: lots of ways — from quick freelancing gigs to building a SaaS
# business — but the right path depends on your skills, time, and risk
# tolerance.
#
# ...
#
# --------------------> ENTERING ask_another_question
# Would you like to ask one more question (yes/no)?


# =====================================================
# Summary
# =====================================================

# In this lesson, we expanded our first LangGraph by adding conditional
# routing.
#
# We:
#
# - Created an ask_question node.
# - Created a chatbot node.
# - Created an ask_another_question node.
# - Created a routing function.
# - Connected the nodes with regular edges.
# - Added a conditional edge after ask_another_question.
# - Routed the user back to ask_question when the response is "yes".
# - Routed the graph to END for any other response.
# - Compiled and invoked the graph.
#
# The graph can now loop based on the value stored in its state.