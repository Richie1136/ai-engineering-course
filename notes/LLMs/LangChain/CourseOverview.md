# Course Overview

## Introduction

We'll begin with a brief discussion of OpenAI's tokens, models, and pricing.

The amount of text we send to a large language model is measured in **tokens**, and OpenAI prices its models based on token usage.

Because of this, it's important to understand the different models offered by OpenAI and the pricing associated with each one.

---

## Environment Setup

In the next section, we'll prepare our Anaconda environment for the course.

We'll also learn how to:

- Obtain an OpenAI API key
- Set the API key as an environment variable

---

## OpenAI API Fundamentals

After setting up our environment, we'll become familiar with the basics of the OpenAI API.

We'll also introduce important terminology related to chat prompting.

Understanding these concepts will be helpful later when we explore LangChain's integration with OpenAI.

---

## LangChain Framework

Next, we arrive at the core of the course—the LangChain library.

We'll begin by discussing the LangChain framework and introducing its main components:

- Model Input
- Model Output
- Chatbot Memory
- Document Retrieval
- Agent Tooling
- LangChain Expression Language (LCEL)

We'll also introduce the protocol that makes implementing all of these components possible.

---

## Model Inputs

We'll begin exploring each component by focusing on model inputs.

Topics include:

- Chat messages
- Chat prompt templates
- Few-shot prompting

This section focuses on asking a chatbot questions and obtaining the desired responses.

---

## Chatbot Memory

Next, we'll explore the classes that enable the creation of a **stateful chatbot**.

A stateful chatbot remembers previous interactions or the ongoing conversation, allowing it to maintain context across multiple exchanges.

---

## Model Outputs

We'll then move on to model outputs.

More specifically, we'll learn how to convert an LLM's response into different output formats, including:

- Strings
- Lists
- DateTime objects

This is useful when the response from a language model needs to be passed into another tool or application that requires a specific data type.

---

## LangChain Expression Language (LCEL)

After covering model inputs and outputs, we'll move on to the LangChain Expression Language (LCEL).

Recall that LCEL is the protocol used for implementing LLM-powered applications.

This is an important topic because it provides the foundation needed for the remainder of the course.

---

## Retrieval-Augmented Generation (RAG)

Next, we'll study **Retrieval-Augmented Generation (RAG)**.

RAG is a technique that allows us to provide a language model with custom data that it was not originally trained on.

This enables the model to generate context-specific answers.

To demonstrate RAG, we'll use the transcripts from the 365 course library to build a simple **365 Q&A chatbot**.

---

## Tools and Agents

The final section of the course covers **Tools** and **Agents**.

Tools give a language model access to the outside world, allowing it to:

- Browse the internet
- Execute code
- Solve mathematical problems

An **Agent** is responsible for selecting the appropriate tools for a task and determining the order in which they should be used.

This part of LangChain gives a chatbot the ability to reason.