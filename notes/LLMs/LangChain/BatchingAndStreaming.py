from dotenv import load_dotenv
import os
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# =====================================================
# Overview
# =====================================================

# In the previous lesson, we saw how easily chains can be created using
# the LangChain Expression Language.
#
# LCEL uses a pipe symbol to connect components.
#
# Connecting components means that the output of one component becomes
# the input of the next.
#
# We also saw that we can call invoke() on individual components, which
# makes the process easier to inspect and understand.
#
# In this lesson, we'll look at two additional methods:
#
# - batch()
# - stream()


# =====================================================
# Load the API Key
# =====================================================

# Load the environment variables from the .env file.

load_dotenv()

# Retrieve the OpenAI API key.

api_key = os.getenv("OPENAI_API_KEY")


# =====================================================
# Initialize the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365,
    max_completion_tokens=700
)


# =====================================================
# Create the Chat Prompt Template
# =====================================================

# This template includes two input variables:
#
# - pet
# - breed

chat_template = ChatPromptTemplate.from_messages([
    (
        "human",
        "I've recently adopted a {pet} which is a {breed}. "
        "Could you suggest several training tips?"
    )
])


# =====================================================
# Create the Chain
# =====================================================

# Create the chain by piping the chat template into the chat model.

chain = chat_template | chat


# =====================================================
# Invoke the Chain
# =====================================================

# invoke() accepts one set of input variables.

# chain.invoke({
#     "pet": "dog",
#     "breed": "shepherd"
# })

# print(
#     chain.invoke({
#         "pet": "dog",
#         "breed": "shepherd"
#     })
# )

# The result is an AIMessage whose content contains training advice.


# =====================================================
# Batching
# =====================================================

# Suppose we need training tips for more than one pet.

# invoke() does not allow us to pass several independent sets of inputs
# at the same time.

# We could call invoke() separately for each input, but LangChain provides
# the batch() method for this purpose.

# batch() runs multiple invoke operations in parallel, making it more
# time-efficient than running each request sequentially.


# =====================================================
# Measure Batch Execution Time
# =====================================================

start_time = time.perf_counter()

# chain.batch([
#     {
#         "pet": "dog",
#         "breed": "Shepherd"
#     },
#     {
#         "pet": "dragon",
#         "breed": "Nightfury"
#     }
# ])

end_time = time.perf_counter()

# print(f"Batch time: {end_time - start_time:.2f} seconds")

# Batch time: 28.93 seconds


# =====================================================
# View the Batch Results
# =====================================================

# print(
#     chain.batch([
#         {
#             "pet": "dog",
#             "breed": "Shepherd"
#         },
#         {
#             "pet": "dragon",
#             "breed": "Nightfury"
#         }
#     ])
# )

# The batch call returns a list containing two AIMessage objects:
#
# - The first contains Shepherd training advice.
# - The second contains fictional Nightfury training advice.
#
# Each message also contains response and token-usage metadata.


# =====================================================
# Measure Sequential Invoke Time
# =====================================================

# First input.

start_time = time.perf_counter()

# chain.invoke({
#     "pet": "dog",
#     "breed": "shepherd"
# })

end_time = time.perf_counter()

# print(
#     f"Invoke time first set: "
#     f"{end_time - start_time:.2f} seconds"
# )

# Invoke time first set: 15.68 seconds


# Second input.

start_time = time.perf_counter()

# chain.invoke({
#     "pet": "dragon",
#     "breed": "night fury"
# })

end_time = time.perf_counter()

# print(
#     f"Invoke time second set: "
#     f"{end_time - start_time:.2f} seconds"
# )

# Invoke time second set: 29.97 seconds


# =====================================================
# Compare Batch and Sequential Execution
# =====================================================

# If batch() genuinely executes requests concurrently, we expect its
# execution time to be shorter than the combined time of the sequential
# invoke() calls.

# Since this is a regular Python file, we can use time.perf_counter()
# to compare the execution times.

# In this example, the batch execution time is less than the combined
# execution time of the two invoke() calls.

# This shows that passing several inputs to batch() can be more
# time-efficient than invoking the chain multiple times in sequence.

# This difference can become even more significant when working with
# more inputs and longer responses.


# =====================================================
# Streaming
# =====================================================

# stream() is another method included in the LangChain Expression
# Language standard interface.

# chain.stream({
#     "pet": "dog",
#     "breed": "shepherd"
# })

# print(
#     chain.stream({
#         "pet": "dog",
#         "breed": "shepherd"
#     })
# )

# Example:
#
# <generator object RunnableSequence.stream at 0x13819e5c0>


# =====================================================
# Generators
# =====================================================

# We encountered generators previously when working with OpenAI's API.

# Generator functions behave like iterators, allowing us to loop over
# their output.

# Unlike standard functions, generators preserve their local state.

# Each subsequent iteration continues from where the previous one left
# off instead of starting again from the beginning.

# This is achieved using yield rather than return.

# A return statement exits the function completely, while yield produces
# values one at a time.

# Generators are also single-use iterators.

# Once a generator has been fully iterated over, it cannot be reset or
# reused.

# This makes generators efficient in terms of memory usage and useful
# for tasks such as streaming responses from a language model.


# =====================================================
# Stream the Model Response
# =====================================================

response = chain.stream({
    "pet": "dog",
    "breed": "shepherd"
})

# Each item returned by the generator is an AIMessageChunk.

# We can extract the content from each chunk and print it continuously.

for chunk in response:
    print(
        chunk.content,
        end=""
    )


# =====================================================
# Streaming Output
# =====================================================

# The response is displayed progressively rather than waiting for the
# entire model response to finish first.

# This allows the text to stream in the same way we typically see when
# communicating with a chatbot.


# =====================================================
# Summary
# =====================================================

# We've laid the foundation for working with the LangChain Expression
# Language.

# We learned how to:
#
# - Pipe components together to form chains.
# - Use invoke() with a single input.
# - Use batch() with multiple inputs.
# - Use stream() to return responses progressively.
#
# Next, we'll explore the technical structure of LCEL chains and the
# types of objects that act as their building blocks.