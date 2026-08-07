from openai import OpenAI
import config

# =============================================================================
# OpenAI Client
# =============================================================================

client = OpenAI(api_key=config.api_key)

# =============================================================================
# Generating Text
# =============================================================================

# This function generates text using an OpenAI GPT model.
#
# Parameters:
# - prompt: The text prompt provided to the model.
# - max_output_tokens: The maximum number of tokens the model can generate.

def generate_text(prompt, max_output_tokens=16):
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        max_output_tokens=max_output_tokens
    )

    return response.output_text.strip()


# =============================================================================
# Basic Example
# =============================================================================

prompt = "Once upon a time"

generated_text = generate_text(prompt)

print(prompt, generated_text)

# Example Output:
# Once upon a time, there was a girl named Fern who loved...

# =============================================================================
# Customizing the Output
# =============================================================================

# The max_output_tokens parameter controls the maximum number of tokens
# returned by the model.
#
# Increasing this value generally allows the model to produce longer responses.

generated_text = generate_text(prompt, 5)

print(prompt, generated_text)

# Example Output:
# Once upon a time...

generated_text = generate_text(prompt, 20)

print(prompt, generated_text)

# Example Output:
# Once upon a time, there was a young boy named Zack...

generated_text = generate_text(prompt, 50)

print(prompt, generated_text)

# Example Output:
# Once upon a time, there was a little girl who was born
# with a very special gift...
