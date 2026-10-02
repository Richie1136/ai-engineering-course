# LangGraph Fundamentals

## Building Conversational AI Memory with LangGraph

### What Is LangGraph?

At its core, LangGraph is a framework that helps you build stateful, multi-step applications powered by language models such as OpenAI's GPT or Anthropic's Claude.

LangGraph allows you to build flows such as conversations or processes that involve:

- Decisions
- Memory
- Loops

These flows are built using a graph-based structure.

You can think of each step in a conversation as a node that connects logically to other nodes, similar to a map.

The result is an agent that doesn't just respond, but can remember and act.

---

## Why Memory Matters

If you've ever built a basic chatbot, you've probably noticed a common limitation:

It doesn't remember what you said a few messages ago.

This happens because language models are stateless.

They don't have built-in memory that automatically retains previous interactions.

This can be inconvenient when we expect chatbots to maintain meaningful and coherent conversations.

LangGraph addresses this limitation by allowing us to manage what is called the graph's **state**.

---

## Graph State

In simple terms, a graph's state is a bundle of information that travels from one node to the next and is updated along the way.

The state can store information such as:

- Previous user inputs
- Tool outputs
- Relevant external data
- Conversation history

This allows the chatbot to remain context-aware and make decisions as the conversation progresses.

---

# Key Building Blocks of a Graph

The main building blocks of a LangGraph application are:

- **State**
- **Nodes**
- **Edges**

---

## State

State holds the data as it flows through the system.

As the graph progresses, the state might contain:

- User input
- Assistant responses
- Conversation history
- Retrieved documents
- Tool outputs

The state is shared and updated as information moves through the graph.

---

## Nodes

Nodes are user-defined Python functions.

Each node performs a specific task in the overall process.

Two special nodes always exist:

- Start node
- End node

These mark where the graph begins and ends.

The remaining nodes are created by the developer to perform meaningful steps in the application.

---

## Example: Employee Onboarding Agent

Imagine an agent that helps onboard new employees and answers their questions.

Suppose an employee named Anna starts the agent.

The start node activates first.

The next node displays a greeting:

> Hey, how can I help you?

Anna asks:

> What's the vacation policy here?

The conversation is then passed to another node.

This node is powered by an LLM and is responsible for answering the question.

It might respond:

> Our vacation policy provides 20 paid days off per year.

The conversation is then passed to another LLM-powered node that summarizes the discussion.

This summary can later be used as context if Anna asks a follow-up question.

Each node performs a focused and straightforward task, making the graph easier to:

- Build
- Test
- Debug

---

## Edges

Edges define how data moves from one node to another.

In a simple graph, the flow may be linear.

Each node points directly to the next node, forming a straight path from the start node to the end node.

However, LangGraph allows more flexible flows.

---

## Conditional Edges

Suppose the final node asks Anna:

> Do you need further assistance?

If Anna responds **yes**, the flow should return to the earlier node where she can ask another question.

If she responds **no**, the graph should continue to the end node.

This is handled using **conditional edges**.

Conditional edges route the flow differently depending on values stored in the state.

For example, the route may depend on whether the user's response is:

- Yes
- No

This branching logic allows us to create dynamic and adaptable applications.

---

# Data Flow Through a Graph

LangGraph uses the state to represent shared and updated data.

As the graph moves from node to node, this data travels with it.

The state might contain:

- User input
- Assistant responses
- Conversation history
- Retrieved documents
- Tool outputs

However, the data needs a defined structure.

This is where schemas are used.

---

## State Schema

A schema defines:

- Which fields are expected in the state
- What data types those fields should contain
- Which fields are optional

The schema can be defined using:

- A Python dictionary, specifically a TypedDict
- A Pydantic model

The difference between state and schema can be thought of like this:

- **State** is the actual data flowing through the graph.
- **Schema** defines the structure and rules that the state should follow.

---

# Memory Management

LangGraph can be used to build chatbots that don't simply react to each message but can also remember previous interactions.

We'll explore two types of memory:

- Short-term memory
- Long-term memory

---

## Short-Term Memory

Short-term memory allows context to persist between graph runs.

This allows the chatbot to remember recent parts of the conversation.

---

## Long-Term Memory

The final lesson of the course introduces long-term memory.

Long-term memory involves storing information externally, such as in:

- A database
- A file

The application can then retrieve that information when necessary.

This allows the chatbot to remember information such as:

- Who the user is
- Previous topics
- Previous goals

---

# Supersteps

In more complex graphs, some nodes may appear horizontally aligned at the same level.

In LangGraph, this level is known as a **superstep**.

A superstep represents a single tick of the graph where multiple nodes can potentially be evaluated and executed in parallel.

Nodes from different supersteps are evaluated sequentially as the graph moves forward step by step.

However, just because nodes belong to the same superstep doesn't mean that all of them will execute.

A node only becomes active if it receives an input state.

Whether a node receives that state depends on the edges leading into it.

Therefore, during a given superstep, only nodes that:

- Receive incoming state from the previous step
- Meet the required edge conditions

will execute.

---

# Course Goal

By the end of this course, you'll have a solid understanding of LangGraph fundamentals.

You'll gain practical experience building:

- Stateful chatbots
- Memory-aware chatbots
- Graph-based conversational applications

You'll also create a working prototype that can later be extended or integrated into real-world applications.