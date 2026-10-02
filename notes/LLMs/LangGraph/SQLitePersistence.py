from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage
)
from langgraph.checkpoint.sqlite import SqliteSaver

from dotenv import load_dotenv

import os
import sqlite3


# =====================================================
# Overview
# =====================================================

# In the previous lessons, we made significant progress by persisting a
# conversation summary to a thread.
#
# This allowed us to invoke the graph multiple times and continue the
# conversation within the same thread.
#
# However, InMemorySaver has one important limitation:
#
# restarting the terminal or stopping the program causes its stored
# checkpoints to be lost.
#
# In this lesson, we'll store our checkpoints in a local SQLite database.
#
# This allows the graph state to be restored even after the program has
# been restarted.
#
# SQLite is a lightweight, disk-based database that works well for
# prototyping.


# =====================================================
# Long-Term Memory with SQLite
# =====================================================

# Instead of storing checkpoints only in memory, we'll use SqliteSaver.
#
# SqliteSaver can be thought of as the persistent counterpart to
# InMemorySaver in this lesson.
#
# Our graph logic will remain the same.
#
# The main difference is where the checkpoints are stored:
#
# InMemorySaver:
#     checkpoints -> memory
#
# SqliteSaver:
#     checkpoints -> SQLite database file


# =====================================================
# Load the API Key
# =====================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


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
#
# This gives the state:
#
# - messages: the current conversation messages
# - summary: the running conversation summary

class State(MessagesState):
    summary: str


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
# It returns:
#
# - An AIMessage containing the question prompt.
# - A HumanMessage containing the user's response.
#
# Since State inherits from MessagesState, the messages key uses the
# add_messages reducer.

def ask_question(
    state: State
) -> State:

    print(
        "\n--------------------> ENTERING ask_question"
    )

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
# The summary is passed through a SystemMessage so that the model can use
# information from previous interactions as context.

def chatbot(
    state: State
) -> State:

    print(
        "\n---------------> ENTERING chatbot"
    )

    print(
        "CHATBOT SUMMARY:",
        state.get("summary", "")
    )

    # Display the current messages so that we can monitor what is being
    # passed through the graph.

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

    # Pass the running summary followed by the current messages to the model.

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
# Create the summarize_and_delete_messages Node
# =====================================================

# This node summarizes the current conversation.
#
# The node:
#
# 1. Converts the current messages into a readable string.
#
# 2. Combines the current conversation with the previous summary.
#
# 3. Asks the model to produce an updated running summary.
#
# 4. Creates RemoveMessage objects for every message currently stored.
#
# 5. Returns the new summary while deleting the current messages.

def summarize_and_delete_messages(
    state: State
) -> State:

    print(
        "STATE SUMMARY:",
        state.get("summary", "")
    )

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

==================== PREVIOUS SUMMARY ====================

{state.get("summary", "")}

==================== NEW CONVERSATION ====================

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
    # Each RemoveMessage contains the ID of the message that should be
    # removed.

    remove_messages = [
        RemoveMessage(
            id=message.id
        )
        for message in state["messages"]
    ]

    # Return:
    #
    # - RemoveMessage objects so the current messages are deleted.
    # - The updated running summary.

    return State(
        messages=remove_messages,
        summary=summary.content
    )


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
    "summarize_and_delete_messages"
)

graph.add_edge(
    "summarize_and_delete_messages",
    END
)


# =====================================================
# Create the SQLite Database
# =====================================================

# Set the path to the SQLite database file.
#
# The database filename ends with .db.
#
# If the database file doesn't already exist, SQLite creates it when the
# connection is established.

db_path = "LangGraph.db"


# =====================================================
# Create the SQLite Connection
# =====================================================

# sqlite3.connect() establishes a connection to the SQLite database.
#
# The database parameter specifies the path to the database file.
#
# check_same_thread controls whether the SQLite connection can only be used
# by the thread that created it.
#
# Setting check_same_thread=False allows the same connection to be used
# across different threads.

con = sqlite3.connect(
    database=db_path,
    check_same_thread=False
)


# =====================================================
# Create the SQLite Checkpointer
# =====================================================

# Create a SqliteSaver and give it the SQLite connection.
#
# SqliteSaver performs the same checkpointing role that InMemorySaver
# performed in the previous lesson, but the checkpoints are stored in the
# SQLite database instead of only in memory.

check_pointer = SqliteSaver(
    conn=con
)


# =====================================================
# Compile the Graph
# =====================================================

# Pass the SQLite checkpointer when compiling the graph.
#
# LangGraph can now save its checkpoints to the SQLite database.

graph_compiled = graph.compile(
    checkpointer=check_pointer
)


# =====================================================
# Display the Graph
# =====================================================

# print(
#     graph_compiled.get_graph().draw_ascii()
# )

#         +-----------+
#         | __start__ |
#         +-----------+
#               *
#               *
#               *
#       +--------------+
#       | ask_question |
#       +--------------+
#               *
#               *
#               *
#          +---------+
#          | chatbot |
#          +---------+
#               *
#               *
#               *
# +-------------------------------+
# | summarize_and_delete_messages |
# +-------------------------------+
#               *
#               *
#               *
#          +---------+
#          | __end__ |
#          +---------+


# =====================================================
# Create a Conversation Thread
# =====================================================

# Each configuration contains a thread_id.
#
# Invocations using the same thread ID belong to the same conversation.
#
# Because the checkpoints are now stored in SQLite, the thread's state can
# be retrieved from the database after restarting the program.

config1 = {
    "configurable": {
        "thread_id": "1"
    }
}


# =====================================================
# Invoke the Graph
# =====================================================

# Pass both the initial state and the thread configuration to invoke().
#
# LangGraph associates the resulting checkpoints with the supplied thread ID.
#
# When the same thread ID is used again, LangGraph can retrieve the state
# previously stored for that thread from the SQLite database.

while True:

    updated_state = graph_compiled.invoke(
        State(),
        config1
    )

    print(updated_state)


# =====================================================
# SQLite Persistence
# =====================================================

# During the first execution, LangGraph stores the checkpoints associated
# with thread 1 inside LangGraph.db.
#
# The program can then be stopped and restarted.
#
# When the graph is invoked again using the same thread ID, SqliteSaver can
# retrieve the previously stored checkpoint data from the database.
#
# The graph therefore has access to the running summary from the previous
# execution.
#
# This is different from InMemorySaver because the checkpoint data isn't
# limited to the lifetime of the Python process.


# =====================================================
# Example
# =====================================================

# Suppose the first execution contains the question:
#
# Where is LeBron James from?
#
# The graph:
#
# 1. Adds the question to messages.
# 2. Generates the chatbot response.
# 3. Creates a running summary.
# 4. Removes the individual messages.
# 5. Saves the resulting state through SqliteSaver.
#
# The program can then be stopped.
#
# When it is restarted and the same thread ID is used, the graph can
# retrieve the saved summary from LangGraph.db.
#
# A follow-up question can therefore use information from the previous
# execution.


# =====================================================
# Summary
# =====================================================

# In this lesson, we replaced InMemorySaver with SqliteSaver.
#
# We:
#
# - Created a local SQLite database file.
# - Established a connection using sqlite3.connect().
# - Set check_same_thread=False.
# - Created a SqliteSaver using the SQLite connection.
# - Passed the SqliteSaver to graph.compile().
# - Continued using a thread ID to identify the conversation.
#
# InMemorySaver stores checkpoints in memory, so its data disappears when
# the process ends.
#
# SqliteSaver stores checkpoints in the SQLite database file, allowing the
# graph to restore its previous state after the program is restarted.