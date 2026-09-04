"""Build and invoke a simple LangChain Expression Language (LCEL) pipeline."""

from dotenv import load_dotenv
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# LCEL composes small components, called Runnables, into a pipeline. Prompts,
# models, parsers, and complete chains share the same execution methods:
#
#     Single input       Multiple inputs       Incremental output
#     invoke()           batch()               stream()
#     ainvoke()          abatch()              astream()
#
# The `a...` variants avoid blocking an asynchronous application while it waits
# for model or network I/O. Because complete chains retain this interface, their
# execution strategy can change without rewriting the pipeline's business logic.


# Configuration
# -----------------------------------------------------------------------------

# Load credentials and configuration from `.env` so secrets stay out of source
# code and the same script can run in different environments.
load_dotenv()


# Components
# -----------------------------------------------------------------------------

# The parser turns model-generated text into predictable Python data. Including
# its formatting instructions in the prompt increases the chance that the model
# will produce text the parser can reliably convert to `list[str]`.
list_output_parser = CommaSeparatedListOutputParser()
list_instructions = list_output_parser.get_format_instructions()

# `from_messages` preserves chat roles. Each tuple contains a role and its message
# template. `{pet}` remains unresolved so the chain can accept different pets.
chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "I've recently adopted a {pet}. Could you suggest three {pet} names?\n"
            + list_instructions,
        )
    ]
)

# Inspect the generated template if needed:
# print(chat_template.messages[0].prompt.template)
#
# Expected template:
#     I've recently adopted a {pet}. Could you suggest three {pet} names?
#     Your response should be a list of comma-separated values ...

# A temperature of zero favors consistency in a teaching example. The seed is an
# additional reproducibility hint, but model output is never guaranteed to match.
chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365,
    max_completion_tokens=1000,
)


# Manual execution
# -----------------------------------------------------------------------------

# Run each stage separately to expose the value crossing each pipeline boundary.
# Prompt templates expect a mapping of variable names to values.
chat_template_result = chat_template.invoke({"pet": "dog"})

# The prompt stage returns concrete chat messages—the input shape expected by a
# chat model. This compatible boundary is what makes composition possible.
# Example: messages=[HumanMessage(content="I've recently adopted a dog ...")]
chat_result = chat.invoke(chat_template_result)

# The model returns an `AIMessage`, not a plain string, so metadata such as token
# usage and tool calls remains available to later stages.
# Example: AIMessage(content="Milo, Luna, Cooper", ...)

# The parser extracts the message content and converts it to a Python list.
# print(list_output_parser.invoke(chat_result))
# Example: ["Milo", "Luna", "Cooper"]


# LCEL pipeline
# -----------------------------------------------------------------------------

# The `|` operator sends each stage's output into the next stage. The resulting
# chain remains a Runnable, so the entire pipeline supports invoke, stream,
# batch, and their asynchronous variants.
chain = chat_template | chat | list_output_parser

# Data flow:
#     {"pet": "dog"} -> prompt value -> AIMessage -> list[str]
chain.invoke({"pet": "dog"})

# print(chain.invoke({"pet": "dog"}))
# Example: ["Milo", "Luna", "Cooper"]
