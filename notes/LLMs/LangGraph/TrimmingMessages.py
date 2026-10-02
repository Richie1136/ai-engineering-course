from langgraph.graph import START, END, StateGraph, add_messages, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal

import os


# =====================================================
# Overview
# =====================================================

# In the previous lessons, we used MessagesState and add_messages() to keep
# appending new messages to the graph state.
#
# This gives the chatbot access to the conversation history and allows it to
# remember earlier parts of the interaction.
#
# However, keeping the entire conversation history becomes increasingly
# expensive as the conversation grows.
#
# More messages means:
#
# - More tokens
# - Longer model inputs
# - More processing time
#
# In this lesson, we'll introduce a new node called trim_messages.
#
# Its purpose is to remove older messages and keep only the final five before
# the user asks another question.


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
# Because MessagesState uses add_messages as its reducer, these new messages
# are appended to the existing state.

def ask_question(state: MessagesState) -> MessagesState:

    print("\n--------------------> ENTERING ask_question")

    for message in state["messages"]:
        message.pretty_print()

    question = "What is your question?"

    print(question)

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

# The chatbot node receives the complete message history currently stored in
# the state.
#
# It passes that history to the chat model and appends the resulting
# AIMessage to the state.

def chatbot(state: MessagesState) -> MessagesState:

    print("\n---------------> ENTERING chatbot")

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

# This node displays the conversation and asks whether the user wants to ask
# another question.
#
# It returns:
#
# - An AIMessage containing the question
# - A HumanMessage containing the user's response

def ask_another_question(
    state: MessagesState
) -> MessagesState:

    print(
        "\n------------> ENTERING ask_another_question"
    )

    for message in state["messages"]:
        message.pretty_print()

    question = (
        "Would you like to ask one more question (yes/no)?"
    )

    print(question)

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
# Create the trim_messages Node
# =====================================================

# Like the other nodes, trim_messages receives a MessagesState object and
# returns an updated MessagesState.
#
# This node removes all messages except the final five.
#
# state["messages"][:-5] selects every message except the final five.
#
# For each old message, we create a RemoveMessage using its ID.
#
# MessagesState uses add_messages as its reducer, so returning these
# RemoveMessage objects causes the matching messages to be removed from the
# state.

def trim_messages(
    state: MessagesState
) -> MessagesState:

    print("\n-----------> ENTERING trim_messages")

    remove_messages = [
        RemoveMessage(
            id=message.id
        )
        for message in state["messages"][:-5]
    ]

    return MessagesState(
        messages=remove_messages
    )


# =====================================================
# Create the Routing Function
# =====================================================

# If the user wants to ask another question, the graph should first route
# through trim_messages.
#
# This removes the older conversation history before returning to
# ask_question.
#
# If the user answers anything other than "yes", the graph ends.

def routing_function(
    state: MessagesState
) -> Literal["trim_messages", "__end__"]:

    if state["messages"][-1].content == "yes":
        return "trim_messages"

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

graph.add_node(
    "trim_messages",
    trim_messages
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

# After trimming the old messages, return to ask_question.

graph.add_edge(
    "trim_messages",
    "ask_question"
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

#                      +-----------+
#                      | __start__ |
#                      +-----------+
#                             *
#                             *
#                             *
#                     +--------------+
#                     | ask_question |
#                     +--------------+
#                    ***            ***
#                  **                  ***
#                **                       **
#       +---------+                         **
#       | chatbot |                          *
#       +---------+                          *
#             *                              *
#             *                              *
#             *                              *
# +----------------------+                   *
# | ask_another_question |                   *
# +----------------------+.                  *
#             .            ......            *
#             .                  .....       *
#             .                       ...    *
#       +---------+                 +---------------+
#       | __end__ |                 | trim_messages |
#       +---------+                 +---------------+


# =====================================================
# Invoke the Graph
# =====================================================

graph_compiled.invoke(
    MessagesState(
        messages=[]
    )
)


# =====================================================
# How the Graph Behaves
# =====================================================

# After entering a question:
#
# 1. We enter ask_question.
# 2. The user's question is added to the state.
# 3. We enter chatbot.
# 4. The chatbot's response is added to the state.
# 5. We enter ask_another_question.
#
# If the user answers "yes":
#
# 6. The graph enters trim_messages.
# 7. All but the final five messages are removed.
# 8. The graph returns to ask_question.
#
# This means the chatbot keeps recent context without preserving the entire
# conversation.


# =====================================================
# Why Keep Five Messages?
# =====================================================

# The final five messages represent the most recent interaction:
#
# - "What is your question?"
# - The user's question
# - The chatbot's answer
# - "Would you like to ask one more question?"
# - "Yes"
#
# This allows the chatbot to keep enough recent context to understand a
# follow-up question.


# =====================================================
# Example: Remembering Recent Context
# =====================================================

# Suppose the user first asks:
#
# Could you give me a grook by Piet Hein?
#
# The chatbot responds.
#
# The user then asks:
#
# Where was the poet born?
#
# Because the recent messages still contain the reference to Piet Hein, the
# chatbot can understand who "the poet" refers to.


# =====================================================
# Another Follow-Up Question
# =====================================================

# Suppose the user later asks:
#
# What else did he do?
#
# If the recent context still contains the previous reference to Piet Hein,
# the chatbot can understand who "he" refers to and continue the conversation.


# =====================================================
# Limitation of Trimming Messages
# =====================================================

# Trimming messages works well when the important subject is mentioned
# frequently enough to remain inside the recent message window.
#
# However, older details will eventually be removed.
#
# For example, if the user asks:
#
# Do you remember the grook you told me?
#
# after that part of the conversation has already been trimmed, the chatbot
# may no longer remember it.
#
# This demonstrates the main tradeoff of message trimming:
#
# - It reduces token usage.
# - It preserves recent context.
# - But older conversation details can be lost.


# =====================================================
# Summary
# =====================================================

# In this lesson, we added a trim_messages node to our LangGraph application.
#
# The node:
#
# - Receives the current MessagesState.
# - Selects every message except the final five.
# - Creates RemoveMessage objects using their IDs.
# - Returns those RemoveMessage objects to the state reducer.
# - Causes add_messages to remove the older messages.
#
# The routing function now sends the graph through trim_messages whenever
# the user wants to continue the conversation.
#
# This creates a middle ground between:
#
# - Sending only the latest message
# - Sending the entire conversation history
#
# The chatbot can retain recent context while preventing the state from
# growing indefinitely.
#
# The disadvantage is that important older conversation details may
# eventually be lost.
#
# The next technique addresses this problem by summarizing older messages.