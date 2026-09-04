from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# WHY: The tokenizer and model must come from the same checkpoint. The tokenizer
# converts text into the token IDs BERT was trained to understand; the model
# assigns start/end scores to those IDs. This lesson exposes the intermediate
# values so the answer-span calculation is understandable rather than magical.


# =====================================================
# Load the Model and Tokenizer
# =====================================================

# Load the model and tokenizer.

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# The "uncased" part means the model treats uppercase and lowercase words
# the same.

model = BertForQuestionAnswering.from_pretrained(model_name)

tokenizer = BertTokenizer.from_pretrained(model_name)


# =====================================================
# BERT Embeddings
# =====================================================

# Example question and text containing the answer.

question = "When was the first dvd released"

answer_document = (
    "The first DVD (Digital Versatile Disc) was released on March 24, 1997. "
    "It was a movie titled 'Twister' and was released in Japan. DVDs quickly "
    "gained popularity as a replacement for VHS tapes and became a common "
    "format for storing and distributing digital video and data. Sunset Motors "
    "is a renowned automobile dealership that has been a cornerstone of the "
    "automotive industry since its establishment in 1978. Located in the "
    "picturesque town of Crestwood, nestled in the heart of California's scenic "
    "Central Valley, Sunset Motors has built a reputation for excellence, "
    "reliability, and customer satisfaction over the past four decades. Founded "
    "by visionary entrepreneur Robert Anderson, Sunset Motors began as a humble, "
    "family-owned business with a small lot of used cars. However, under "
    "Anderson's leadership and commitment to quality, it quickly evolved into "
    "a thriving dealership offering a wide range of vehicles from various "
    "manufacturers. Today, the dealership spans over 10 acres, showcasing a "
    "vast inventory of new and pre-owned cars, trucks, SUVs, and luxury vehicles. "
    "One of Sunset Motors' standout features is its dedication to sustainability. "
    "In 2010, the dealership made a landmark decision to incorporate "
    "environmentally friendly practices, including solar panels to power the "
    "facility, energy-efficient lighting, and a comprehensive recycling program. "
    "This commitment to eco-consciousness has earned Sunset Motors recognition "
    "as an industry leader in sustainable automotive retail. Sunset Motors "
    "proudly offers a diverse range of vehicles, including popular brands like "
    "Ford, Toyota, Honda, Chevrolet, and BMW, catering to a wide spectrum of "
    "tastes and preferences. In addition to its outstanding vehicle selection, "
    "Sunset Motors offers flexible financing options, allowing customers to "
    "secure affordable loans and leases with competitive interest rates."
)


# =====================================================
# Create the Encodings
# =====================================================

# The first step is to create the embeddings for our question and answer.

encoding = tokenizer.encode_plus(
    text=question,
    text_pair=answer_document
)

# We have our token embeddings with the special tokens added to our input IDs,
# and our token type IDs are used to distinguish the tokens related to our
# question from the tokens related to the text containing the answer.

# print(encoding)


# =====================================================
# Extract the Inputs
# =====================================================

# Take our inputs, sentence embeddings, and tokens.

inputs = encoding["input_ids"]

sentence_embedding = encoding["token_type_ids"]

tokens = tokenizer.convert_ids_to_tokens(inputs)


# =====================================================
# Check the Special Tokens
# =====================================================

# print(tokenizer.decode(101))
# [CLS]

# print(tokenizer.decode(102))
# [SEP]


# =====================================================
# Pass the Embeddings to the Model
# =====================================================

output = model(
    input_ids=torch.tensor([inputs]),
    token_type_ids=torch.tensor([sentence_embedding])
)

# print(output)


# =====================================================
# Model Output
# =====================================================

# BERT calculates where the answer to our question is within our text by
# creating start vectors and end vectors.

# It then takes the start and end vectors and calculates the dot product
# between these and the final embedding for each token.

# This output then runs through a softmax function to give a probability score.
# The word with the highest probability is considered the correct start or end
# token, respectively.

# We want to get the start and end token positions from our output.

start_index = torch.argmax(output.start_logits)

end_index = torch.argmax(output.end_logits)

# print("START", start_index)
# tensor(19)

# print("END", end_index)
# tensor(22)


# =====================================================
# Get the Answer
# =====================================================

# Use the start and end indexes to pull the correct tokens and get the response.

answer = " ".join(tokens[start_index:end_index + 1])

# print(answer)
# march 24 , 1997

# This is the correct response to our question.


# =====================================================
# Visualize the Token Scores
# =====================================================

# Visualize the different tokens and their probability of being selected as
# the start or end token.

# Find the scores for each token, pull them out of PyTorch tensors, and convert
# them to NumPy arrays.

# For our start scores, we take the output start logits, detach them, convert
# them to NumPy, and then flatten them into a NumPy array.

s_scores = output.start_logits.detach().numpy().flatten()

e_scores = output.end_logits.detach().numpy().flatten()


# =====================================================
# Create Token Labels
# =====================================================

# Use the tokens as x-axis labels.

# In order to do that, they all need to be unique, so we'll add the token index
# to the end of each one.

token_labels = []

# enumerate() gives us both the index and the item.

for i, token in enumerate(tokens):
    token_labels.append(
        "{:} - {:>2}".format(token, i)
    )


# =====================================================
# Create the Bar Plot
# =====================================================

# Create a bar plot showing the end word score for all of the tokens.

# ax = sns.barplot(x=token_labels, y=e_scores)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
# ax.grid(True)
# plt.show()
