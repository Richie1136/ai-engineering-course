from dotenv import load_dotenv

from langchain_classic.output_parsers import DatetimeOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI


# =====================================================
# Datetime Output Parser
# =====================================================

# A model normally returns dates as text, and that text can take many forms:
#
#     January 1, 2027
#     01/01/2027
#     2027-01-01
#
# Those formats are readable to people but unreliable for a program. A
# DatetimeOutputParser asks the model for one precise format and converts the
# response into a Python datetime object.
#
# Mental model:
# parser instructions -> prompt -> formatted model text -> parser -> datetime


# =====================================================
# 1. Load Environment Variables
# =====================================================

# Load OPENAI_API_KEY from the local .env file. This keeps the secret outside
# source code while allowing ChatOpenAI to read it from the environment.
load_dotenv()


# =====================================================
# 2. Initialize the Chat Model
# =====================================================

chat = ChatOpenAI(
    model="gpt-5-mini",
    # Structured output should be predictable, so we reduce randomness.
    temperature=0,
    seed=365,
)


# =====================================================
# 3. Create the Datetime Parser
# =====================================================

# The same parser has two responsibilities:
#
# 1. Provide instructions describing the required response format.
# 2. Convert a correctly formatted response into datetime.datetime.
datetime_output_parser = DatetimeOutputParser()

format_instructions = datetime_output_parser.get_format_instructions()


# =====================================================
# 4. Build a Prompt with the Format Instructions
# =====================================================

# The instructions must be part of the message sent to the model. Creating
# them without inserting them into the prompt would not affect the response.
message_human = HumanMessage(
    content=f"""
When was the Danish poet Piet Hein born?

{format_instructions}
""".strip()
)

# Printing the prompt while learning makes it clear that the parser's format
# instructions are actually sent to the model.
print("Prompt sent to the model:")
print(message_human.content)


# =====================================================
# 5. Ask the Model for a Formatted Response
# =====================================================

# ChatOpenAI returns an AIMessage. Its content should contain only the datetime
# string requested by the parser instructions.
response = chat.invoke([message_human])

print("\nRaw model response:")
print(response.content)

# Expected response shape:
# 1905-12-16T00:00:00.000000Z

# =====================================================
# 6. Parse the Text into a Python Datetime
# =====================================================

# invoke() reads the text stored in the AIMessage and validates it against the
# required format. If the format is correct, it returns a datetime object.
response_parsed = datetime_output_parser.invoke(response)

print("\nParsed Python value:")
print(response_parsed)

# Expected Python type:
# <class 'datetime.datetime'>

# Once parsed, the program can safely access date components or perform date
# arithmetic rather than manually interpreting an unpredictable string.
print("\nParsed value type:")
print(type(response_parsed))

print("\nIndividual date components:")
print("Year:", response_parsed.year)
print("Month:", response_parsed.month)
print("Day:", response_parsed.day)

# =====================================================
# Summary
# =====================================================

# get_format_instructions() handles the REQUEST side by telling the model which
# datetime format to produce.
#
# invoke(response) handles the CONVERSION side by validating the returned text
# and converting it to datetime.datetime.
#
# This parser is useful when model-generated dates will be sorted, compared,
# stored, or used in calculations by the rest of an application.
