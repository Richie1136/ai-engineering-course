from dotenv import load_dotenv
import os
from langchain_openai.chat_models import ChatOpenAI

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

# ChatOpenAI is LangChain's wrapper around OpenAI chat models.
#
# model:
#     Specifies which OpenAI model to use.
#
# model_kwargs:
#     Allows additional model-specific parameters to be passed to the
#     underlying OpenAI API. Here we provide a seed so responses are
#     more reproducible.
#
# temperature:
#     Controls the randomness of the model's responses.
#     Lower values produce more deterministic outputs.

chat = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    seed=365
)


# =====================================================
# Invoke the Model
# =====================================================

# invoke() is the primary method used to interact with a LangChain chat model.
#
# It accepts several input types, including a simple string prompt.
#
# Throughout the course you'll often see prompts written using triple
# quotation marks. This makes long prompts easier to read and allows
# quotation marks to appear inside the prompt without escaping them.

response = chat.invoke(
    "I've recently adopted a dog. Could you suggest some dog names?"
)


# =====================================================
# Display the Response
# =====================================================

# invoke() returns an AIMessage object rather than a plain string.
#
# The generated text is stored inside the content attribute.

# print(response.content)

# Congrats on your new dog! Here are name ideas organized by type to help you find one that fits.

# Short/training-friendly (1–2 syllables)
# - Max, Scout, Ace, Finn, Jax, Milo, Bear, Duke, Tess, Nala

# Female
# - Bella, Luna, Daisy, Molly, Sadie, Ruby, Penny, Willow, Rosie, Stella

# Male
# - Charlie, Cooper, Buddy, Rocky, Teddy, Leo, Oscar, Rex, Hank, Jasper

# Unisex
# - Bailey, Riley, Pepper, Scout, Frankie, River, Marley, Sky, Dakota, Blue

# Nature-inspired
# - Willow, River, Aspen, Maple, Sage, Storm, Sunny, Ivy, Cedar, Ocean

# Unique / quirky
# - Pixel, Nimbus, Quill, Sprocket, Echo, Juno, Koda, Nyx, Zephyr, Miso

# Food-inspired
# - Peanut, Mochi, Cinnamon, Honey, Biscuit, Pumpkin, Chip, Cocoa

# Pop-culture / literary
# - Yoda, Loki, Arya, Simba, Chewie, Neo, Eleven, Frodo, Zelda, Leia

# Quick tips for choosing a name
# - Keep it short and 1–2 syllables if you plan on training often (easier for recall).
# - Avoid names that sound like common commands (e.g., “Kit” ~ “sit”).
# - Say it aloud a few times—make sure you like calling it in public.
# - Try the name for a few days and see how your dog responds; you can always adjust.

# If you tell me your dog’s sex, breed/size, color, or personality, I can suggest names tailored to them. Which details would you like to share?