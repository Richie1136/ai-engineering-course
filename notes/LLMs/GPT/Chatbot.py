from openai import OpenAI
import config

# =============================================================================
# OpenAI Client
# =============================================================================

client = OpenAI(api_key=config.api_key)

# =============================================================================
# Poetic Chatbot
# =============================================================================

# This example demonstrates how a system message can be used to control
# the behavior and personality of a chatbot.
#
# In this example, the chatbot responds to every question in the form of poetry.
# We also provide a few example conversations (few-shot prompting) so the
# model learns the desired response style.

def poetic_chatbot(prompt):
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=[
            {
                "role": "system",
                "content": "You are a poetic chatbot."
            },
            {
                "role": "user",
                "content": "When was Google founded?"
            },
            {
                "role": "assistant",
                "content": (
                    "In the late '90s, a spark did ignite, "
                    "Google emerged, a radiant light. "
                    "By Larry and Sergey, in '98, it was born, "
                    "a search engine new, on the web it was sworn."
                )
            },
            {
                "role": "user",
                "content": "Which country has the youngest president?"
            },
            {
                "role": "assistant",
                "content": (
                    "Ah, the pursuit of youth in politics, a theme we explore. "
                    "In Austria, Sebastian Kurz did implore, "
                    "at the age of 31, his journey did begin, "
                    "leading with vigor, in a world filled with din."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_output_tokens=256
    )

    return response.output_text.strip()


# =============================================================================
# Example
# =============================================================================

prompt = "When was cheese first made?"

response = poetic_chatbot(prompt)

print(response)

# Example Output:
#
# Oh, cheese, delightful creation from ancient times,
# Let me share its story in lilting rhymes.
# From the land of Mesopotamia,
# Around 6000 BCE came savory glee.
# A timeless treasure with flavors untold,
# Enjoyed by the young and cherished by the old.