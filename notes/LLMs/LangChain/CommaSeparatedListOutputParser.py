from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_openai.chat_models import ChatOpenAI


# =====================================================
# Why This Lesson Matters
# =====================================================

# A language model normally returns free-form text. That is convenient for a
# person to read, but awkward for a program to use. For example, an application
# cannot reliably loop over dog names if the response also contains headings,
# explanations, and bullet points.
#
# An output parser helps us turn the model's text into a predictable Python
# type. Here, CommaSeparatedListOutputParser turns this text:
#
#     Max, Luna, Charlie
#
# into this Python value:
#
#     ["Max", "Luna", "Charlie"]
#
# There are TWO related jobs in this process:
#
# 1. Tell the model which text format to produce.
# 2. Parse that formatted text into a Python object.
#
# The parser helps with both jobs, but it cannot reliably repair a response
# that ignored the requested format. This is why its format instructions must
# be included in the prompt before we call the model.


# =====================================================
# 1. Load Environment Variables
# =====================================================

# ChatOpenAI needs an API key to authenticate with OpenAI. load_dotenv() reads
# OPENAI_API_KEY from the local .env file and places it in the environment.
# We do not put the actual key in this file because source code may be shared or
# committed to Git.
load_dotenv()


# =====================================================
# 2. Initialize the Chat Model
# =====================================================

# This object is the LangChain interface to the OpenAI chat model.
chat = ChatOpenAI(
    model="gpt-5-mini",
    # A temperature of 0 reduces randomness. Predictable responses are useful
    # when another piece of code expects a specific output format.
    temperature=0,
    # The seed makes repeated examples more reproducible when the model and
    # API support seeded generation. It is not an absolute guarantee that every
    # response will be identical.
    seed=365,
)


# =====================================================
# 3. Create the Output Parser
# =====================================================

# We create this object before the prompt because it provides the exact
# formatting instruction that the model should follow. Later, the same object
# will convert the response into a list.
list_output_parser = CommaSeparatedListOutputParser()

format_instructions = list_output_parser.get_format_instructions()

# The returned instruction is similar to:
# "Your response should be a list of comma separated values..."
# Keeping this instruction in a variable lets us insert it into the prompt.


# =====================================================
# 4. Build a Prompt That Requests Parseable Output
# =====================================================

# An f-string inserts the parser's instructions into the message sent to the
# model. In the earlier version of this example, get_format_instructions() was
# evaluated by itself, which did nothing useful because its result was never
# included in the prompt.
message_human = HumanMessage(
    content=f"""
I've recently adopted a dog. Suggest ten dog names.

{format_instructions}
""".strip()
)

# Printing the message is optional. It is helpful while learning because it
# proves that both our request and the format instructions are present.
# print(message_human.content)

# We see that the instructions have been successfully added to the message.

# I've recently adopted a dog. Could you suggest some dog names?


# =====================================================
# 5. Ask the Model for a Response
# =====================================================

# invoke() sends the ordered message list to the model. The result is an
# AIMessage object containing generated text plus metadata such as token usage.
response = chat.invoke([message_human])

print(response.content)


# Expected response.content shape:
# Milo, Luna, Charlie, Ruby, Finn, Willow, Baxter, Nala, Cooper, Olive


# =====================================================
# 6. Convert the Response to a Python List
# =====================================================

# The parser reads AIMessage.content, splits the comma-separated values, and
# returns list[str]. After this step, normal Python code can loop over, sort, or
# validate the names without having to understand a paragraph of prose.
response_parsed = list_output_parser.invoke(response)

print(response_parsed)

# We retrieve a list with each name as a separate element.

# Expected parsed value:
# ['Milo', 'Luna', 'Charlie', 'Ruby', 'Finn', 'Willow', 'Baxter', 'Nala', 
# 'Cooper', 'Olive']


# =====================================================
# Summary / Mental Model
# =====================================================

# Model output begins as text. If an application needs structured data:
#
# parser instructions -> prompt -> model text -> parser -> Python value
#
# get_format_instructions() handles the REQUEST side: it tells the model how to
# write its response.
#
# invoke(response) handles the CONVERSION side: it changes the returned text
# into a list.
#
# Both sides matter. Parsing unstructured prose by commas can create nonsense
# list items because normal sentences also contain commas.
