from openai import OpenAI
import config

# =============================================================================
# OpenAI Client
# =============================================================================

client = OpenAI(api_key=config.api_key)

# =============================================================================
# Keyword Extraction
# =============================================================================

# This example creates a function that accepts a large block of text and
# extracts its most important keywords and phrases.
#
# Because the model can understand language and context, it can identify
# important words and phrases within the provided text.
#
# The messages below provide instructions and examples that demonstrate
# how the model should respond.

# =============================================================================
# Message Roles
# =============================================================================

# System message:
# Provides instructions describing what the model should do and how it should
# respond.

# User message:
# Represents the input provided by the user. In this example, the user provides
# a block of text from which keywords should be extracted.

# Assistant message:
# Provides an example of the expected response to the preceding user message.
# These examples help demonstrate the desired output format.


def text_summarizer(prompt):
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You will be provided with a block of text, and your task "
                    "is to extract a list of keywords and key phrases from it."
                )
            },
            {
                "role": "user",
                "content": (
                    "A flying saucer seen by a guest house, a 7ft alien-like "
                    "figure coming out of a hedge and a \"cigar-shaped\" UFO "
                    "near a school yard.\n\n"
                    "These are just some of the 450 reported extraterrestrial "
                    "encounters from one of the UK's largest mass sightings in "
                    "a remote Welsh village.\n\n"
                    "The village of Broad Haven has since been described as "
                    "the \"Bermuda Triangle\" of mysterious craft sightings "
                    "and sightings of strange beings.\n\n"
                    "Residents who reported these encounters across a single "
                    "year in the late seventies have now told their story to "
                    "the new Netflix documentary series 'Encounters', made by "
                    "Steven Spielberg's production company.\n\n"
                    "It all happened back in 1977, when the Cold War was at "
                    "its height and Star Wars and Close Encounters of the "
                    "Third Kind dominated the box office."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "flying saucer, guest house, 7ft alien-like figure, hedge, "
                    "cigar-shaped UFO, school yard, extraterrestrial encounters, "
                    "UK, mass sightings, remote Welsh village, Broad Haven, "
                    "Bermuda Triangle, mysterious craft sightings, strange beings, "
                    "residents, late seventies, Netflix documentary series, "
                    "Steven Spielberg, production company, 1977, Cold War, "
                    "Star Wars, Close Encounters of the Third Kind, "
                    "science fiction blockbuster, box office"
                )
            },
            {
                "role": "user",
                "content": (
                    "Each April, in the village of Maeliya in northwest "
                    "Sri Lanka, Pinchal Weldurelage Siriwardene gathers his "
                    "community under the shade of a large banyan tree. "
                    "The tree overlooks a human-made body of water called a "
                    "wewa, meaning reservoir or \"tank\" in Sinhala. "
                    "The wewa stretches beside the village's rice paddies for "
                    "175 acres and is filled with rainwater from preceding "
                    "months.\n\n"
                    "Siriwardene, the 76-year-old secretary of the village's "
                    "agrarian committee, performs a tightly guarded ritual. "
                    "By boiling coconut milk on an open hearth beside the tank, "
                    "he seeks blessings for a prosperous harvest from the "
                    "deities residing in the tree.\n\n"
                    "By releasing water into irrigation canals, the tank "
                    "supports the rice crop during the dry months before the "
                    "rains arrive. For nearly two millennia, lake-like water "
                    "bodies such as this have helped generations of farmers "
                    "cultivate their fields."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "April, Maeliya, northwest Sri Lanka, "
                    "Pinchal Weldurelage Siriwardene, banyan tree, wewa, "
                    "reservoir, tank, Sinhala, rice paddies, 175 acres, "
                    "rainwater, agrarian committee, coconut milk, open hearth, "
                    "blessings, prosperous harvest, deities, irrigation canals, "
                    "dry months, farmers, cultivate"
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

prompt = (
    "Master Reef Guide Kirsty Whitman didn't need to tell me twice. "
    "Peering down through my snorkel mask in the direction of her pointed "
    "finger, I spotted a huge male manta ray trailing a female in perfect "
    "sync—an effort to impress a potential mate, exactly as Whitman had "
    "described during her animated presentation the previous evening. "
    "Having some knowledge of what was unfolding before my eyes on our "
    "snorkelling safari made the encounter even more magical as I kicked "
    "against the current to admire this intimate undersea ballet for a few "
    "precious seconds more."
)

keywords = text_summarizer(prompt)

print(keywords)

# Example Output:
# Master Reef Guide, Kirsty Whitman, snorkel mask, male manta ray,
# female manta ray, potential mate, animated presentation,
# snorkelling safari, undersea ballet