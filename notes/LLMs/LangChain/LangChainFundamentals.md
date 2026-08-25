# Build Chat Applications with OpenAI and LangChain

## Large Language Models

Large language models (LLMs) are models that can process human language and generate human-like responses.

The **large** part of the name refers to the enormous amount of data these models have been trained on and, consequently, their impressive number of parameters.

But how large is large?

As training data grows and models become more knowledgeable, the meaning of the word **large** continues to adapt.

OpenAI's GPT-2 model, released in 2019, was reported to have 1.5 billion parameters and was trained on 40 GB of text.

These are impressively large numbers, but compare them to GPT-3, which was released a year later.

GPT-3 contains 175 billion parameters and was trained on 570 GB of data, roughly the equivalent of 400 billion tokens.

Even those are small numbers compared to the size and training of GPT-4.

The word **large** is therefore not anchored to specific numbers. Instead, it is an umbrella term for models of this scale.

Researchers in the field could always take inspiration from the creative names astronomers give to their telescopes, such as large models, very large models, and extremely large models. However, given the rate at which new and more capable models are released, we would quickly run out of adjectives.

---

## Chat Models

A subset of large language models is explicitly trained to hold conversations.

These are intuitively called **chat models**.

An example is GPT, which stands for **Generative Pre-trained Transformer**.

If you've ever had a conversation with ChatGPT, then you've encountered OpenAI's chat model capabilities.

This course will use GPT-4 to create custom chatbots.

---

## LangChain

LangChain is a framework that allows for the seamless development of applications powered by large language models.

Throughout the course, our main focus will be creating chatbots that are:

- **Stateful**
- **Context-aware**
- **Capable of reasoning**

### Stateful

Being stateful refers to keeping memories of past or ongoing conversations.

### Context-Aware

Context awareness refers to answering questions about information outside the model's training.

### Reasoning

Reasoning refers to the ability to choose among various tools and determine their order of execution to solve a specific task.

---

## Chatbot Use Cases

The use cases for these types of chatbots are endless.

Some examples include:

- Using chatbots in web browsers to summarize the most important news of the day.
- Querying a database using everyday language.
- Building a multi-source fact checker.
- Integrating a chatbot into the 365 platform to answer course-related questions.

---

## 365 Q&A Chatbot

Once we solidify our knowledge of the LangChain library, we'll demonstrate how to build a **365 Q&A chatbot**.

# LangChain Fundamentals

## Overview

LangChain is a framework for building **LLM-powered applications**.

Its primary goal is to make it easier to develop applications that are:

- Stateful
- Context-aware
- Capable of reasoning

To simplify this process, LangChain divides an application into several reusable components, with each component responsible for a specific task.

---

# The Three Main Components

LangChain organizes its functionality into three primary modules:

1. Model I/O
2. Retrieval
3. Agent Tooling

These components are connected through the **LangChain Expression Language (LCEL)**.

---

# Model I/O

The first part of an LLM-powered application is **Model I/O**.

Model I/O consists of everything involved in sending information to a language model and processing its response.

It includes:

- Large Language Models (LLMs)
- Prompts
- Prompt Templates
- Example Selectors
- Output Parsers

---

## Prompts

A prompt is simply the text we send to the language model.

For example:

> I've recently adopted a dog. Could you suggest some dog names?

The model uses this prompt to generate its response.

---

## Prompt Templates

Prompt templates make prompts reusable.

Instead of writing a completely new prompt every time, we can replace parts of the prompt with variables.

For example:

```text
I've recently adopted a {pet}. Could you suggest some names?
```

If we provide:

```
pet = "dog"
```

the model generates dog names.

If we instead provide:

```
pet = "cat"
```

the exact same prompt template generates cat names.

Prompt templates make it much easier to build reusable AI applications.

---

## Few-Shot Prompting

Prompt templates are especially useful when performing **few-shot prompting**.

Rather than simply telling the model what to do, we provide several examples demonstrating the desired behavior.

The model then uses those examples to produce responses in the same format.

---

## Output Parsers

The response returned by a language model is normally an assistant message containing plain text.

However, many applications require the output in a different format.

For example:

- A Python application may require a Pandas DataFrame.
- Another language model may expect XML input.
- An application may require a list or a date.

Output parsers convert the model's response into the required format.

Common output parsers include:

- String Output Parser
- DateTime Output Parser
- Comma-Separated List Output Parser

Together, the Model I/O module consists of:

- Prompt
- Language Model
- Output Parser

Example selectors are also part of this module and help choose appropriate examples for few-shot prompting.

---

# Retrieval

The second major LangChain component is the **Retrieval** module.

Retrieval makes chatbots **context-aware**.

Instead of relying only on the model's training data, we can provide external or proprietary information that the model can use when answering questions.

This technique allows the chatbot to answer questions about information it was never originally trained on.

The Retrieval module consists of several components:

- Document Loader
- Text Splitter
- Embedding Model
- Vector Store
- Retriever

These components work together to retrieve relevant information before sending it to the language model.

In this course, we'll build a chatbot capable of answering questions about the **365 Introduction to Data and Data Science** course.

---

# Agent Tooling

The third major module is **Agent Tooling**.

Agent Tooling enables the creation of **reasoning chatbots**.

Instead of simply generating text, these chatbots can decide:

- Which tool to use
- When to use it
- In what order multiple tools should be executed

Individual tools can also be grouped together into **toolkits** for solving specific types of tasks.

---

# LangChain Expression Language (LCEL)

The three major modules are connected through the **LangChain Expression Language (LCEL)**.

LCEL is the protocol that links together:

- Model I/O
- Retrieval
- Agent Tooling

It forms a significant portion of this course and is best understood through practical examples.

---

# Templates

LangChain also provides a collection of pre-built templates.

These are pre-implemented prompt templates that have already been designed for common AI tasks.

Using these templates can significantly speed up development.

---

# LangSmith

LangSmith is responsible for the **observability** of an application.

It allows developers to inspect, monitor, and evaluate the behavior of their LLM-powered applications.

---

# LangServe

LangServe is responsible for **deployment**.

It allows LangChain applications to be deployed as APIs that other applications can access.

---

# Documentation

Two resources are especially useful when learning LangChain.

## LangChain Documentation

The official documentation contains:

- Tutorials
- How-to guides
- Conceptual explanations
- Integration guides

It is an excellent resource that you'll likely revisit many times throughout your development journey.

---

## API Reference

The API Reference documents every:

- Function
- Class
- Method

This resource is invaluable when writing LangChain applications and will be referenced frequently throughout this course.

---

# Key Takeaways

- LangChain simplifies the development of LLM-powered applications.
- It is designed to build applications that are **stateful**, **context-aware**, and capable of **reasoning**.
- The framework is organized into three primary modules:
  - Model I/O
  - Retrieval
  - Agent Tooling
- Prompt templates make prompts reusable through variables.
- Output parsers convert model responses into application-friendly formats.
- Retrieval enables chatbots to answer questions using external data.
- Agent Tooling allows chatbots to select and use tools autonomously.
- LCEL connects the major LangChain components together.
- LangSmith provides monitoring and observability.
- LangServe is used to deploy LangChain applications.
- The official documentation and API Reference are essential learning resources.