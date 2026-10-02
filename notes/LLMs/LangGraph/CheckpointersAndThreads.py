from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage
)
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv

import os


# =====================================================
# Overview
# =====================================================

# In this section, we'll learn how to persist conversations so that our
# graph can retain memory across multiple graph invocations.
#
# In LangGraph, this is made possible through a checkpointing system.
#
# Recall that a graph consists of nodes connected by direct or conditional
# edges.
#
# Nodes on different horizontal levels are executed sequentially as the
# graph progresses.
#
# Nodes on the same horizontal level belong to the same superstep.
#
# Depending on the graph's edges and conditions, multiple nodes within a
# superstep may become active.


# =====================================================
# Checkpointers
# =====================================================

# A checkpointer in LangGraph saves the graph's state as it progresses
# through execution.
#
# The saved state and its associated metadata are represented through
# StateSnapshot objects.
#
# These snapshots make it possible to inspect previously saved graph states.
#
# Checkpoints can also support workflows where a human inspects or modifies
# state during execution.
#
# This can be useful for testing, debugging, and manually directing an agent.


# =====================================================
# Threads
# =====================================================

# A thread is a collection of related checkpoints.
#
# Each thread is identified by a thread ID.
#
# The thread ID allows LangGraph to distinguish between different
# conversations.
#
# When the graph is invoked again using the same thread ID, LangGraph can
# restore the state associated with that thread and continue the
# conversation.
#
# Threads therefore provide continuity between graph invocations.


# =====================================================
# Short-Term and Long-Term Memory
# =====================================================

# In this section, we'll explore two types of memory:
#
# - Short-term memory
# - Long-term memory
#
# Short-term memory, also called thread-scoped memory, remembers interactions
# within a particular thread.
#
# When using InMemorySaver, this information is stored in memory and is
# therefore volatile.
#
# Restarting the notebook kernel or shutting down the process removes this
# in-memory state.
#
# Long-term memory persists beyond the lifetime of a single session.
#
# This type of memory can use external storage rather than relying entirely
# on information stored in memory.
#
# Relevant past information can then be retrieved when needed for future
# conversations.


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

    # Store this prompt and the user's input in the graph state as an
    # AIMessage and HumanMessage.

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
# Create the Checkpointer
# =====================================================

# Create an instance of InMemorySaver.
#
# This acts as the graph's short-term checkpoint storage.
#
# It allows LangGraph to save the state associated with each thread while
# the current process remains active.

check_pointer = InMemorySaver()


# =====================================================
# Compile the Graph
# =====================================================

# Pass the checkpointer when compiling the graph so that LangGraph can save
# checkpoints during execution.

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
# Create Conversation Threads
# =====================================================

# Now that we've created the checkpointing mechanism, we need thread IDs
# that allow LangGraph to associate checkpoints with particular
# conversations.
#
# Each configuration contains a thread_id.
#
# Invocations using the same thread ID belong to the same conversation.
#
# Invocations using different thread IDs belong to separate conversations.

config1 = {
    "configurable": {
        "thread_id": "1"
    }
}

config2 = {
    "configurable": {
        "thread_id": "2"
    }
}


# =====================================================
# Invoke the Graph
# =====================================================

# Pass both the initial state and the thread configuration to invoke().
#
# LangGraph associates the resulting checkpoints with the supplied thread ID.
#
# When the same thread ID is used again, the graph can retrieve the state
# previously saved for that thread.

while True:

    updated_state = graph_compiled.invoke(
        State(),
        config2
    )

    print(updated_state)


# =====================================================
# How Thread Persistence Works
# =====================================================

# During the first invocation of a thread, the summary is initially empty.
#
# The user asks a question, the chatbot responds, and
# summarize_and_delete_messages creates a running summary.
#
# The messages are then removed, leaving the summary in the graph state.
#
# Because the graph was compiled with a checkpointer, this resulting state
# is associated with the thread ID.
#
# When the graph is invoked again using the same thread ID, the saved state
# from that conversation is available again.
#
# This means the chatbot can continue using the running summary from the
# previous graph invocation.


# =====================================================
# Multiple Threads
# =====================================================

# One benefit of thread persistence is that we can maintain multiple
# independent conversations.
#
# For example:
#
# config1 uses thread_id "1".
# config2 uses thread_id "2".
#
# Each thread maintains its own checkpoints.
#
# Therefore, starting a conversation with thread 2 does not require using
# the conversation state associated with thread 1.


# =====================================================
# Current Limitation
# =====================================================

# InMemorySaver stores checkpoints in memory.
#
# This means it can preserve state across multiple graph invocations while
# the current process is running.
#
# However, restarting the notebook kernel or shutting down the process
# removes this in-memory checkpoint data.
#
# Memory that must survive beyond the current process requires persistent
# storage.


# =====================================================
# Summary
# =====================================================

# In this lesson, we introduced LangGraph checkpointing and threads.
#
# We learned that:
#
# - A checkpointer saves graph state during execution.
# - Saved graph states are represented through checkpoints.
# - Related checkpoints can be grouped into threads.
# - Each thread is identified using a thread_id.
# - Reusing the same thread ID allows a conversation to continue from its
#   previously saved state.
# - Different thread IDs allow separate conversations to maintain separate
#   state.
# - InMemorySaver provides in-memory checkpoint storage.
# - In-memory checkpoint data is lost when the process is restarted.
#
# Combining checkpointing with our running conversation summary allows the
# chatbot to preserve context across separate graph invocations within the
# same thread.