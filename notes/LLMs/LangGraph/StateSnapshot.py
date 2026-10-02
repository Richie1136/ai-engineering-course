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

# In the previous lesson, we introduced checkpointers and threads.
#
# A checkpointer allows LangGraph to save the graph's state as it progresses
# through execution.
#
# Related checkpoints are grouped together using a thread ID.
#
# In this lesson, we'll examine these checkpoints in more detail using
# StateSnapshot objects.
#
# We'll use get_state_history() to retrieve the checkpoints associated with
# a thread and inspect how the graph state changes throughout execution.


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
# Create the Checkpointer
# =====================================================

# Create an instance of InMemorySaver.
#
# This provides in-memory checkpoint storage for the graph.

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
        config1
    )

    print(updated_state)


    # =====================================================
    # Retrieve the State History
    # =====================================================

    # Now let's examine the checkpoints in more detail.
    #
    # get_state_history() retrieves the state history associated with a
    # particular thread.
    #
    # The method accepts the thread configuration as its argument.
    #
    # get_state_history() returns a generator object.

    graph_states = [
        state
        for state in graph_compiled.get_state_history(config1)
    ]


    # =====================================================
    # Inspect the StateSnapshot Objects
    # =====================================================

    # Each item returned from get_state_history() is a StateSnapshot.
    #
    # A StateSnapshot contains information about the graph at a particular
    # point in its execution.
    #
    # This includes information such as:
    #
    # - The state values
    # - The next node to execute
    # - Metadata
    # - Thread information
    # - Checkpoint information
    #
    # The first item returned by get_state_history() is the most recent
    # snapshot.
    #
    # Reverse the list using [::-1] so that we can inspect the checkpoints
    # from the beginning of the conversation to the end.

    for state in graph_states[::-1]:

        print(
            f"""
Messages: {state.values["messages"]}
Summary: {state.values.get("summary", "")}
Next: {state.next}
Step: {state.metadata["step"]}
"""
        )


# =====================================================
# Understanding the StateSnapshots
# =====================================================

# Let's examine how the snapshots change throughout one graph invocation.


# =====================================================
# Step -1: Initial Input
# =====================================================

# The first snapshot represents the state that was initially passed into
# the graph.
#
# At this point:
#
# - messages is empty.
# - summary doesn't exist yet.
# - The graph has not entered its first node.
# - The next node is __start__.
#
# The step number is -1 because graph execution has not yet begun.

# Messages: []
# Summary:
# Next: ('__start__',)
# Step: -1


# =====================================================
# Step 0: After START
# =====================================================

# The START node doesn't modify our graph state.
#
# The state therefore still contains:
#
# - An empty messages list.
# - No conversation summary.
#
# The next node is ask_question.
#
# This is where the normal step count begins.

# Messages: []
# Summary:
# Next: ('ask_question',)
# Step: 0


# =====================================================
# Step 1: After ask_question
# =====================================================

# After ask_question executes, the messages list contains:
#
# - One AIMessage asking for the user's question.
# - One HumanMessage containing the user's question.
#
# The next node is chatbot.

# Messages:
#
# [
#     AIMessage(
#         content='What is your question?',
#         ...
#     ),
#     HumanMessage(
#         content='Where is lebron james from?',
#         ...
#     )
# ]
#
# Summary:
#
# Next: ('chatbot',)
# Step: 1


# =====================================================
# Step 2: After chatbot
# =====================================================

# After the chatbot node executes, the model's response is appended to the
# messages list.
#
# The messages list now contains:
#
# - The AIMessage asking for the question.
# - The HumanMessage containing the user's question.
# - The AIMessage containing the chatbot's response.
#
# The next node is summarize_and_delete_messages.

# Messages:
#
# [
#     AIMessage(
#         content='What is your question?',
#         ...
#     ),
#     HumanMessage(
#         content='Where is lebron james from?',
#         ...
#     ),
#     AIMessage(
#         content='LeBron James was born in Akron, Ohio...',
#         ...
#     )
# ]
#
# Summary:
#
# Next: ('summarize_and_delete_messages',)
# Step: 2


# =====================================================
# Step 3: After summarize_and_delete_messages
# =====================================================

# summarize_and_delete_messages:
#
# - Creates a running summary.
# - Removes all messages from the state.
#
# Therefore, messages is now empty and summary contains the information from
# the conversation.
#
# There are no additional nodes to execute, so next is an empty tuple.

# Messages: []
#
# Summary:
# - The user asked where LeBron James is from.
# - The assistant replied that LeBron James was born in Akron, Ohio,
#   grew up there, and attended St. Vincent-St. Mary High School.
#
# Next: ()
# Step: 3


# =====================================================
# Additional Invocations
# =====================================================

# From the next invocation onward, the same sequence repeats.
#
# Because config1 uses the same thread ID, the previous checkpointed state
# is available to the graph.
#
# The running summary can therefore continue to be updated as new questions
# are asked.


# =====================================================
# Full StateSnapshot Object
# =====================================================

# Printing the StateSnapshot object itself provides considerably more
# information than the simplified output above.
#
# For example:

# print(graph_states[0])

# A StateSnapshot contains information such as:
#
# - values
# - next
# - config
# - metadata
# - created_at
# - parent_config
# - tasks
# - interrupts
#
# The values field contains the actual graph state.
#
# The next field identifies which node or nodes are scheduled next.
#
# The config field contains checkpoint and thread configuration.
#
# The metadata field contains information such as the graph step.
#
# parent_config connects the checkpoint to the checkpoint that came before
# it.
#
# tasks contains information related to the graph tasks associated with
# that snapshot.


# =====================================================
# StateSnapshot Progression
# =====================================================

# The checkpoints from one invocation can be viewed as:
#
# Step -1
#
# Input state
#     ↓
#
# Step 0
#
# START
#     ↓
#
# Step 1
#
# ask_question
#     ↓
#
# Step 2
#
# chatbot
#     ↓
#
# Step 3
#
# summarize_and_delete_messages
#     ↓
#
# END
#
# Each snapshot allows us to inspect what the graph state looked like at a
# particular point during execution.


# =====================================================
# Summary
# =====================================================

# In this lesson, we examined LangGraph checkpoints in more detail using
# StateSnapshot objects.
#
# We learned that:
#
# - get_state_history() retrieves the checkpoint history for a thread.
#
# - get_state_history() returns a generator.
#
# - Each item produced by the generator is a StateSnapshot.
#
# - The snapshots are returned with the most recent snapshot first.
#
# - Reversing the snapshots lets us inspect graph execution from beginning
#   to end.
#
# - StateSnapshot.values contains the graph state at that checkpoint.
#
# - StateSnapshot.next identifies the next node or nodes to execute.
#
# - StateSnapshot.metadata contains additional checkpoint information,
#   including the graph step.
#
# By examining each snapshot, we can follow how the state changes as the
# graph progresses from its initial input through each node and finally
# reaches the end of the graph.