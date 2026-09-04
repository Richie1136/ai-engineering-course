from langchain_core.prompts import PromptTemplate

# WHY: Hard-coded prompts are difficult to reuse. A PromptTemplate marks the
# changing pieces as variables, validates that values are supplied, and creates
# a PromptValue compatible with other LangChain components.


# =====================================================
# Prompt Templates and Prompt Values
# =====================================================

# We've explored and tested LangChain's ChatOpenAI class.

# We examined the invoke() method and saw that, when used with a ChatOpenAI
# object, it accepts either:
#
# - A string
# - A list of chat messages

# The chat messages we explored included:
#
# - SystemMessage: stores instructions for the model
# - HumanMessage: stores the user's question or request
# - AIMessage: stores model responses or example responses

# These message objects are useful, but they have an important limitation:
# they are not reusable.

# LangChain therefore provides prompt abstractions called prompt templates.

# Before moving into chat prompt templates, it's helpful to first understand
# prompt templates created from simple strings rather than chat messages.


# =====================================================
# Create a String Template
# =====================================================

TEMPLATE = """
System:
{description}

Human:
I've recently adopted a {pet}.
Could you suggest some {pet} names?
"""

# The values inside curly braces are placeholders.
#
# In this example:
#
# - {description} represents the description of the chatbot's behavior
# - {pet} represents the type of pet


# =====================================================
# Create a PromptTemplate
# =====================================================

# Use the from_template() method to create a PromptTemplate object from the
# string template defined above.

prompt_template = PromptTemplate.from_template(
    template=TEMPLATE
)

# print(type(prompt_template))

# <class 'langchain_core.prompts.prompt.PromptTemplate'>

# The object is an instance of the PromptTemplate class.

# print(prompt_template)

# The input_variables parameter stores the variables that were defined inside
# the template.

# In this example, those variables are:
#
# - description
# - pet

# The original string template is also stored inside the PromptTemplate object.


# =====================================================
# Invoke the PromptTemplate
# =====================================================

# PromptTemplate implements the familiar invoke() method.

# The invoke() method expects a dictionary containing values for each of the
# input variables.

prompt_value = prompt_template.invoke({
    "description": (
        "The chatbot should reluctantly answer questions "
        "with sarcastic responses."
    ),
    "pet": "dog"
})

# print(prompt_value)


# =====================================================
# Prompt Value
# =====================================================

# print(type(prompt_value))

# <class 'langchain_core.prompt_values.StringPromptValue'>

# The result is an instance of the StringPromptValue class.

# In contrast:
#
# - ChatOpenAI.invoke() accepts a string or list of chat messages and returns
#   a chat message object.
#
# - PromptTemplate.invoke() accepts a dictionary and returns a prompt value
#   object.


# =====================================================
# View the Filled-In Template
# =====================================================

# The text attribute stores the completed prompt.

# print(prompt_value.text)

# The description placeholder has been replaced with the chatbot description,
# and the pet placeholder has been replaced with the word "dog".

# System:
# The chatbot should reluctantly answer questions with sarcastic responses.
#
# Human:
# I've recently adopted a dog.
# Could you suggest some dog names?


# =====================================================
# Process Summary
# =====================================================

# First, we created a string template containing input variables defined
# inside curly braces.

# Second, we created a PromptTemplate instance using that string.

# Third, we called invoke() and provided values for both input variables.

# This produced a StringPromptValue containing the completed template.

# We'll also see that ChatOpenAI's invoke() method can accept prompt value
# objects in addition to strings and lists of chat messages.

# This becomes important later in the course when we connect objects together
# and use the output of one object as the input of another.
