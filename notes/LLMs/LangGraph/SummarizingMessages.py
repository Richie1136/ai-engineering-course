from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage
)

from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal

import os


# =====================================================
# Overview
# =====================================================

# In the previous lesson, we learned how to reduce the number of messages
# stored in the graph state.
#
# We removed messages by returning RemoveMessage objects to the
# add_messages reducer.
#
# We'll now implement another approach for long-running conversations.
#
# Instead of storing previous interactions exactly as they occurred, we'll
# summarize them.
#
# Our plan is:
#
# 1. Create a custom State that inherits from MessagesState.
#
# 2. Add a new summary key that stores the running conversation summary.
#
# 3. Modify the chatbot node so that it receives both the current messages
#    and the existing summary.
#
# 4. Replace the trim_messages node with a node that updates the summary
#    and then deletes the messages currently stored in the state.


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
# Define the State
# =====================================================

# Create a custom State that inherits the messages key and its reducer
# from MessagesState.
#
# Add an additional key called summary.
#
# Unlike the messages key, summary does not use a reducer.
#
# Therefore, whenever summary is updated, its previous value is replaced
# with the new value.

class State(MessagesState):
    summary: str


# =====================================================
# Test the State
# =====================================================

test_state = State()

# Attempting to access a missing key directly would raise a KeyError.
#
# print(test_state["summary"])
#
# This would become a problem when we need to check whether a conversation
# summary already exists.
#
# Instead, we can use get().
#
# get() accepts:
#
# 1. The key name
# 2. A default value to return if the key doesn't exist
#
# An empty string works well as the default because:
#
# - An empty string evaluates to False.
# - A populated string evaluates to True.

# print(test_state.get("summary", ""))


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

# The ask_question node asks the user for a new question.
#
# The function returns:
#
# - An AIMessage containing the question prompt.
# - A HumanMessage containing the user's response.
#
# Since State inherits from MessagesState, the messages key still uses
# add_messages as its reducer.

def ask_question(
    state: State
) -> State:

    print(
        "\n--------------------> ENTERING ask_question"
    )

    # We removed the message pretty_print() loop from this node so that the
    # console output doesn't become unnecessarily cluttered.

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

# The chatbot node receives:
#
# - The current messages stored in the state.
# - The running conversation summary.
#
# The summary is added as a SystemMessage so that the model can use the
# earlier conversation as context even though those original messages may
# already have been deleted.

def chatbot(
    state: State
) -> State:

    print(
        "\n---------------> ENTERING chatbot"
    )

    # Display the current messages so that we can monitor what is actually
    # being passed through the graph.

    for message in state["messages"]:
        message.pretty_print()

    # Retrieve the existing summary.
    #
    # If a summary has not yet been created, get() returns an empty string.

    system_message = f"""
Here's a quick summary of what's been discussed so far:

{state.get("summary", "")}

Keep this in mind as you answer the next question.
"""

    # Pass the summary as a SystemMessage followed by the current messages.

    response = chat.invoke(
        [
            SystemMessage(
                system_message
            )
        ]
        + state["messages"]
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

# This node asks whether the user wants to continue the conversation.
#
# It returns:
#
# - An AIMessage containing the question.
# - A HumanMessage containing the user's answer.

def ask_another_question(
    state: State
) -> State:

    print(
        "\n------------> ENTERING ask_another_question"
    )

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
# Create the summarize_and_delete_messages Node
# =====================================================

# Instead of keeping only the final five messages, this node summarizes
# the entire current conversation.
#
# The node:
#
# 1. Converts the messages into a readable string.
#
# 2. Combines the current conversation with the previous summary.
#
# 3. Asks the model to produce an updated running summary.
#
# 4. Creates RemoveMessage objects for every message currently stored.
#
# 5. Returns the new summary while deleting all current messages.

def summarize_and_delete_messages(
    state: State
) -> State:

    print(
        "\n-----------> ENTERING summarize_and_delete_messages"
    )

    # Store the current conversation in a readable string.

    new_conversation = ""

    # Append each message's type and content.

    for message in state["messages"]:
        new_conversation += (
            f"{message.type}: "
            f"{message.content}\n\n"
        )

    # Create instructions for updating the running summary.
    #
    # If no previous summary exists, an empty string is inserted.

    summary_instructions = f"""
Update the ongoing summary by incorporating the new lines of conversation below.

Build upon the previous summary rather than repeating it so that the result
reflects the most recent context and developments.

Previous Summary:

{state.get("summary", "")}

New Conversation:

{new_conversation}
"""

    print(summary_instructions)

    # Ask the model to create the updated summary.

    summary = chat.invoke(
        [
            HumanMessage(
                summary_instructions
            )
        ]
    )

    # Delete every message currently stored in the state.
    #
    # Each RemoveMessage uses the ID of the message it should remove.

    remove_messages = [
        RemoveMessage(
            id=message.id
        )
        for message in state["messages"]
    ]

    # Return:
    #
    # - RemoveMessage objects so the old messages are deleted.
    # - The updated running summary.

    return State(
        messages=remove_messages,
        summary=summary.content
    )


# =====================================================
# Create the Routing Function
# =====================================================

# If the user wants to continue, route the graph through
# summarize_and_delete_messages.
#
# This updates the running summary and removes the current messages.
#
# If the user responds with anything other than "yes", the graph ends.

def routing_function(
    state: State
) -> Literal[
    "summarize_and_delete_messages",
    "__end__"
]:

    if state["messages"][-1].content == "yes":
        return "summarize_and_delete_messages"

    return "__end__"


# =====================================================
# Define the Graph
# =====================================================

graph = StateGraph(
    State
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
    "summarize_and_delete_messages",
    summarize_and_delete_messages
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

# After summarizing the conversation and deleting the messages,
# return to ask_question.

graph.add_edge(
    "summarize_and_delete_messages",
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

#                       +-----------+
#                       | __start__ |
#                       +-----------+
#                             *
#                             *
#                             *
#                     +--------------+
#                     | ask_question |
#                     +--------------+
#                     **              ***
#                  ***                   **
#                **                        ***
#       +---------+                           **
#       | chatbot |                            *
#       +---------+                            *
#             *                                *
#             *                                *
#             *                                *
# +----------------------+                     *
# | ask_another_question |                     *
# +----------------------+...                  *
#             .              .....             *
#             .                   ......       *
#             .                         ...    *
#       +---------+           +-------------------------------+
#       | __end__ |           | summarize_and_delete_messages |
#       +---------+           +-------------------------------+


# =====================================================
# Invoke the Graph
# =====================================================

graph_compiled.invoke(
    State(
        messages=[]
    )
)


# =====================================================
# How the Summary Works
# =====================================================

# Suppose the user asks:
#
# Could you give me a grook by Piet Hein?
#
# The chatbot answers.
#
# When the user chooses to continue, the graph enters
# summarize_and_delete_messages.
#
# The current conversation is converted into a readable string and sent to
# the model.
#
# Since there is no previous summary yet, Previous Summary is empty.
#
# The model then creates the first running summary.
#
# Afterward, all current messages are deleted from the state.


# =====================================================
# Updating an Existing Summary
# =====================================================

# Suppose the user next asks:
#
# Where was the poet born?
#
# The chatbot can still determine that "the poet" refers to Piet Hein because
# the earlier discussion is contained inside the running summary.
#
# When the conversation is summarized again:
#
# - The previous summary is included.
# - The new conversation is included.
# - The model produces an updated summary containing both.


# =====================================================
# Preserving Older Context
# =====================================================

# Suppose the user later asks:
#
# What else did he do?
#
# The raw messages from the original discussion may already have been
# removed.
#
# However, the summary still contains enough information to identify
# Piet Hein and understand the follow-up question.


# =====================================================
# Difference from Message Trimming
# =====================================================

# With message trimming, old messages eventually disappear completely.
#
# This can cause the chatbot to forget earlier details.
#
# With summarization, the original messages are still removed, but important
# information from those messages is preserved inside the running summary.
#
# The chatbot receives:
#
# - The running summary
# - The new messages
#
# instead of the entire raw conversation history.


# =====================================================
# Current Memory Limitation
# =====================================================

# We've started integrating memory into the chatbot.
#
# However, the memory implemented so far only exists during the current
# execution.
#
# If the graph is invoked again from the beginning, it won't remember the
# previous conversation.
#
# The state starts fresh with each new invocation.
#
# The next step is thread-level persistence, which allows memory to persist
# across multiple executions within the same conversation thread.


# =====================================================
# Summary
# =====================================================

# In this lesson, we improved the chatbot's memory by summarizing older
# conversation history.
#
# We:
#
# - Created a custom State that inherits from MessagesState.
# - Added a summary key to the state.
# - Used get() to safely retrieve the summary.
# - Passed the running summary to the chatbot through a SystemMessage.
# - Created summarize_and_delete_messages.
# - Converted the current messages into a readable conversation string.
# - Asked the model to update the running summary.
# - Deleted the original messages after summarizing them.
# - Preserved earlier context without keeping the entire raw message history.
#
# This approach reduces the amount of raw conversation history stored in the
# state while still allowing the chatbot to retain important information from
# earlier parts of the conversation.
#
# The remaining limitation is that this memory only lasts for the current
# graph execution.