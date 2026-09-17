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

## 5. Output Parsers

- `StringOutputParser.py` — Uses a string output parser to convert a model's
  `AIMessage` response into plain text that can be passed directly to another
  component or used by the application.
- `CommaSeparatedListOutputParser.py` — Converts comma-separated model output
  into a Python list and provides formatting instructions that can be included
  in the prompt.
- `DatetimeOutputParser.py` — Parses model-generated date and time text into a
  structured Python datetime value.

**Section summary:** Output parsers transform raw model output into a format
that is easier for application code to consume. They can also provide format
instructions that tell the model how its response should be structured.

## 6. LangChain Expression Language (LCEL)

- `LCELPiping.py` — Connects prompts, chat models, and output parsers with the
  LCEL pipe operator (`|`) so the output of one component becomes the input to
  the next.
- `BatchingAndStreaming.py` — Demonstrates processing multiple inputs with
  `batch()` and receiving model output incrementally with `stream()`.
- `RunnableSequence.py` — Explores the `Runnable` interface and
  `RunnableSequence`, which represent components that can be composed into an
  ordered execution pipeline.
- `RunnablePassthrough.py` — Uses `RunnablePassthrough` to preserve incoming
  values while passing data through or adding values needed by later parts of
  a chain.
- `RunnableParallel.py` — Uses `RunnableParallel` to send the same input through
  multiple runnable branches and collect their results.
- `RunnableLambda.py` — Wraps regular Python functions so custom logic can
  participate in an LCEL chain.

**Section summary:** LCEL provides a standard way to compose LangChain
components. Runnables can be chained sequentially, executed in parallel,
batched, streamed, or combined with custom Python logic while keeping the
application pipeline readable and reusable.

## 7. Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) extends an LLM with external knowledge by
retrieving relevant information and providing it to the model as additional
context.

The detailed RAG workflow, concepts, and runnable examples are organized
separately in the `RAG/` folder.

See `RAG/Summary.md` for the complete RAG quick reference covering document
loading, document splitting, embeddings, vector stores, similarity search,
MMR, retrievers, document stuffing, and response generation.

**Section summary:** RAG allows LangChain applications to answer questions using
private, current, or domain-specific information that may not exist in the
model's training data.

## Core Workflows at a Glance

```text
Basic chat:
prompt -> ChatOpenAI.invoke() -> AIMessage -> .content

Reusable prompt:
template + values -> PromptValue -> chat model -> AIMessage

Output parsing:
prompt -> chat model -> AIMessage -> output parser -> usable Python value

LCEL:
prompt | model | output parser
   ↓
input -> component -> component -> component -> output

RunnableParallel:
                     -> runnable A ->
input -> parallel ->                 -> combined output
                     -> runnable B ->
```

For the complete RAG workflow, see `RAG/Summary.md`.

## Terms to Remember

- **Token:** A unit of text processed and billed by a model.
- **Context window:** The maximum tokens available for input and output.
- **Prompt template:** Reusable prompt text containing placeholders.
- **Prompt value:** A completed prompt ready to pass to a model.
- **Output parser:** A component that transforms model output into a more useful
  representation such as a string, list, or datetime.
- **LCEL:** LangChain Expression Language, used to compose components into
  chains where one component's output becomes the next component's input.
- **Runnable:** LangChain's standard interface for components that can be
  invoked, chained, batched, streamed, or composed with other components.
- **RunnableSequence:** A sequence of Runnables executed one after another.
- **RunnablePassthrough:** A Runnable that preserves or forwards input while
  allowing additional values or transformations to be added.
- **RunnableParallel:** A Runnable that executes multiple branches using the
  same input and combines their outputs.
- **RunnableLambda:** A wrapper that turns a Python callable into a Runnable so
  it can participate in an LCEL chain.
- **RAG:** Retrieval-Augmented Generation, a technique that retrieves relevant
  external information and supplies it to a language model as context before
  generating a response.