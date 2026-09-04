# LangChain — Summary and Quick Reference

Use this page as a map of the completed lessons in this section. The Python
files contain runnable examples and detailed inline comments; the Markdown
files contain the supporting concepts.

## 1. Course and Framework Foundations

- `CourseOverview.md` — Roadmap from tokens and the OpenAI API through prompts,
  memory, output parsing, RAG, LCEL, tools, and agents.
- `LangChainFundamentals.md` — The three main LangChain areas: model I/O,
  retrieval, and agent tooling. It also introduces LCEL, LangSmith, LangServe,
  templates, and the official documentation.
- `WhatMakesLangChainPowerful.md` — Why useful LLM applications need more than
  a model: state, external context, reasoning tools, monitoring, and deployment.
- `BusinessApplications.md` — Business examples involving conversation
  summaries, protection of sensitive data, support automation, and agents.

**Section summary:** LangChain is an application framework that connects a
language model to prompts, conversation state, external knowledge, tools, and
operational services.

## 2. Tokens, Models, and Configuration

- `Tokens.md` — What tokens are, how tokenization varies by context, and why
  token counts affect context limits and cost.
- `ModelsAndPrices.md` — Input/output pricing, context windows, and the
  trade-offs involved in selecting a model.
- `SettingAPIKeyAsEnvVariable.py` — Loads `OPENAI_API_KEY` from `.env` without
  placing the secret directly in source code.

**Section summary:** Before calling a model, choose it according to capability,
context size, latency, and cost, and keep credentials outside the codebase.
Token usage is the common unit behind both context limits and API billing.

## 3. Calling Chat Models and Using Messages

- `FirstSteps.py` — Builds foundational OpenAI API calls and explains system,
  user, assistant, and tool roles plus response metadata and common parameters.
- `ChatOpenAI.py` — Wraps an OpenAI chat model with LangChain and calls it with
  `invoke()`.
- `SystemAndHumanMessages.py` — Separates behavior instructions from the user's
  request with `SystemMessage` and `HumanMessage`.
- `AIMessages.py` — Uses `AIMessage` objects as example answers for few-shot
  prompting.

**Section summary:** A chat request is an ordered message sequence. System
messages establish behavior, human messages supply requests, and AI messages
represent generated or example responses. LangChain returns an `AIMessage`, so
the response text is normally read from `.content`.

## 4. Reusable Prompts and Few-Shot Examples

- `PromptTemplatesAndValues.py` — Replaces placeholders in a reusable string
  prompt and produces a `StringPromptValue`.
- `ChatPromptTemplatesAndValues.py` — Combines system and human templates,
  fills their variables, and produces a `ChatPromptValue` for the model.
- `FewShotChatMessagePromptTemplates.py` — Packages example human/AI exchanges
  into a reusable few-shot chat prompt.

**Section summary:** Templates separate prompt structure from runtime data.
The usual flow is `template -> invoke(values) -> prompt value -> model`. Few-shot
templates add demonstrations so the model can imitate a desired format or tone.

## 5. Custom Data and Retrieval

- `LangChain.py` — Explains why general model knowledge is insufficient for
  private, current, or domain-specific questions and outlines retrieval.
- `AddingCustomData.py` — Implements a webpage-based RAG pipeline with a loader,
  text splitter, embeddings, FAISS vector search, question rewriting, retrieved
  context, and conversation history.

**Section summary:** RAG loads source data, splits it into chunks, converts the
chunks to embeddings, stores them in a vector database, retrieves relevant
chunks for a question, and gives those chunks to the model as context.

## Core Workflows at a Glance

```text
Basic chat:
prompt -> ChatOpenAI.invoke() -> AIMessage -> .content

Reusable prompt:
template + values -> PromptValue -> chat model -> AIMessage

RAG:
documents -> chunks -> embeddings -> vector store
question -> retrieval -> relevant context -> chat model -> grounded answer
```

## Terms to Remember

- **Token:** A unit of text processed and billed by a model.
- **Context window:** The maximum tokens available for input and output.
- **Prompt template:** Reusable prompt text containing placeholders.
- **Prompt value:** A completed prompt ready to pass to a model.
- **Embedding:** A numerical representation used to compare semantic meaning.
- **Vector store:** A database optimized for similarity search over embeddings.
- **Retriever:** The component that selects relevant source chunks.
- **RAG:** Generation supported by context retrieved from an external source.
- **LCEL:** LangChain Expression Language, used to compose components into
  chains where one component's output becomes the next component's input.
