# LangGraph — Summary and Quick Reference

## What This Section Covers

This section builds a conversational application step by step: a basic graph,
conditional loops, accumulated message history, trimming and summarization,
and conversation persistence with checkpoints and SQLite.

LangGraph organizes the application into nodes connected by edges. Shared
state carries the information each step needs. The application supplies prior
messages or a summary to the model so it can answer follow-up questions with
context.

## File Guide

- [LangGraphFundamentals.md](LangGraphFundamentals.md) — Introduces state,
  schemas, nodes, edges, conditional routing, memory, and supersteps.
- [FirstGraph.py](FirstGraph.py) — Defines a typed message state and a chatbot
  node, connects `START -> chatbot -> END`, and compiles the graph.
- [ConditionalEdges.py](ConditionalEdges.py) — Adds interactive question nodes
  and a routing function that loops when the user enters `yes`.
- [ReducerFunctions.py](ReducerFunctions.py) — Uses `Annotated` and
  `add_messages` to accumulate messages instead of replacing the history.
- [MessageState.py](MessageState.py) — Uses the predefined `MessagesState`
  schema and introduces message removal by ID.
- [TrimmingMessages.py](TrimmingMessages.py) — Adds a node that removes all but
  the last five messages before the next question.
- [SummarizingMessages.py](SummarizingMessages.py) — Maintains a running
  summary, supplies it to the model, and removes the summarized messages.
- [CheckpointersAndThreads.py](CheckpointersAndThreads.py) — Uses
  `InMemorySaver` and thread IDs to retain state across graph invocations.
- [StateSnapshot.py](StateSnapshot.py) — Inspects checkpoint history,
  including saved values, next nodes, and step metadata.
- [SQLitePersistence.py](SQLitePersistence.py) — Uses `SqliteSaver` with a
  local SQLite database to persist thread checkpoints across process restarts.

## Graph Building Blocks

| Concept | Role in these lessons |
| --- | --- |
| State | Holds messages and, in later examples, a running summary |
| State schema | Defines state fields and types using `TypedDict` or `MessagesState` |
| Node | A Python function that reads state, performs work, and returns updates |
| Edge | Connects one step to the next |
| Conditional edge | Selects the next step using a routing function |
| `START` / `END` | Mark graph entry and termination |
| Reducer | Defines how an update combines with an existing state field |
| Superstep | One execution step in which active nodes may run in parallel |

The basic construction sequence in the examples is:

```text
define state -> define node functions -> create StateGraph
             -> add nodes and edges -> compile -> invoke
```

## Conditional Routing and Loops

The interactive examples follow this flow:

```text
START -> ask_question -> chatbot -> ask_another_question
             ^                              |
             |----------- yes --------------|
                                            |-- otherwise -> END
```

`add_conditional_edges()` connects the decision node to a routing function.
The function reads the user's reply and returns the next node name or
`"__end__"`. These examples compare the reply to the exact string `"yes"`.

The trimming and summarization lessons insert a memory-management node on
the return path before `ask_question`.

## State Updates and Message History

In the initial examples, the `messages` field has no reducer, so each update
replaces its previous value. A conversation loop alone therefore does not
preserve earlier messages.

`ReducerFunctions.py` attaches `add_messages` to the field:

```python
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

Later examples use `MessagesState`, which already provides the message field
and its reducer. Nodes return their new messages, and the reducer merges those
updates into the existing history. `RemoveMessage(id=message.id)` requests
removal of a particular message through that same reducer.

The summarization examples extend the schema:

```python
class State(MessagesState):
    summary: str
```

The `messages` field retains its reducer, while an update to `summary`
replaces the previous summary. `state.get("summary", "")` handles the initial
case where no summary has been created.

## Managing Conversation Context

| Approach | What the model receives | Main tradeoff |
| --- | --- | --- |
| Full history | All accumulated messages | Input size grows with the conversation |
| Trimming | Recent messages retained in state | Older details are lost |
| Summarization | Running summary plus current messages | Compression may omit details and requires an extra model call |

`TrimmingMessages.py` selects `state["messages"][:-5]` and returns removal
requests for those messages. The retained five messages represent one recent
interaction in this example, including the prompts and the continuation reply.
This is a message-count window, not a token budget.

`SummarizingMessages.py` combines the previous summary with the current
conversation, asks the model for an updated summary, and removes all current
messages. The chatbot receives the summary through a `SystemMessage` followed
by the new conversation messages. This preserves compressed context after the
original messages have been removed.

## Checkpointers and Threads

Trimming and summarization control the contents of state. Checkpointing allows
that state to be retained between graph invocations.

The persistence examples compile the graph with a checkpointer and invoke it
with a thread configuration:

```python
check_pointer = InMemorySaver()
graph_compiled = graph.compile(checkpointer=check_pointer)

config = {"configurable": {"thread_id": "1"}}
updated_state = graph_compiled.invoke(State(), config)
```

Reusing the same thread ID with the same checkpoint storage continues that
conversation's saved state. Different thread IDs keep separate conversation
states. In these lessons, each invocation follows:

```text
START -> ask_question -> chatbot -> summarize_and_delete_messages -> END
```

An outer Python loop invokes the graph again. The saved summary supplies the
context for the next question.

| Checkpointer | Storage | Survives a process restart? |
| --- | --- | --- |
| `InMemorySaver` | Process memory | No |
| `SqliteSaver` | SQLite database file | Yes, when the same database is reused |

`SQLitePersistence.py` opens `LangGraph.db` using a relative path, so its
location depends on the working directory from which the script runs.
SQLite makes the examples' thread state durable; the implemented workflow
still retrieves conversation state by thread ID rather than searching for
user memories across different threads.

## Inspecting Saved State

`StateSnapshot.py` calls `graph_compiled.get_state_history(config1)` and
reverses the collected history to print it in chronological order.

The lesson inspects these snapshot fields:

- `values`: Saved state, including `messages` and `summary`.
- `next`: Nodes scheduled to execute next; an empty tuple indicates no next
  nodes for that snapshot.
- `metadata["step"]`: The associated execution step.

Inspecting snapshots shows how messages accumulate, when summarization runs,
and what remains in state after message removal.

## Key Takeaways

- State carries data; nodes perform work; edges determine execution flow.
- Conditional routing enables loops, but message retention depends on the
  state's update behavior.
- `MessagesState` provides the message reducer used by the later examples.
- Trimming limits recent history; summarization preserves compressed context.
- A checkpointer and thread ID provide continuity across graph invocations.
- In-memory checkpoints last for the current process; SQLite checkpoints can
  be reused after restarting it.
- Snapshot history makes changes to conversation state easier to inspect.
