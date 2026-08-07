from transformers import AutoTokenizer


# The Transformers pipeline automatically handles tokenization for us.
# However, we can also work directly with a tokenizer when we want more
# control over the preprocessing step.

# Tokenization is the process of breaking text into smaller pieces, called
# tokens, before passing them to a language model.

# Different language models use different tokenization strategies, so it is
# important to use the tokenizer that matches the model you plan to use.


# =====================================================
# Load a Pre-trained Tokenizer
# =====================================================

# Specify the model we want to use.

# Using the correct tokenizer ensures that the text is tokenized in the way
# expected by the model.

model = "bert-base-uncased"

# Load the pre-trained tokenizer.

tokenizer = AutoTokenizer.from_pretrained(model)


# =====================================================
# Tokenize a Sentence
# =====================================================

sentence = "I'm so excited to be learning about large language models."

# Running the tokenizer returns a dictionary containing several outputs.

input_ids = tokenizer(sentence)

# print(input_ids)

# {
#     'input_ids': [101, 1045, 1005, 1049, 2061, 7568, 2000, 2022,
#                   4083, 2055, 2312, 2653, 4275, 1012, 102],
#     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# }

# Three outputs have been generated:
#
# - input_ids
# - token_type_ids
# - attention_mask

# token_type_ids are used by some models to distinguish one sentence from
# another. For example, tokens belonging to the first sentence may receive
# a value of 0, while tokens belonging to the second sentence receive a
# value of 1.

# The attention_mask tells the model which tokens should be attended to when
# processing the input.


# =====================================================
# View the Tokens
# =====================================================

# The tokenizer first breaks the sentence into individual tokens.

tokens = tokenizer.tokenize(sentence)

# print(tokens)

# ['i', "'", 'm', 'so', 'excited', 'to', 'be', 'learning',
#  'about', 'large', 'language', 'models', '.']

# The sentence has now been converted into tokens.


# =====================================================
# Convert Tokens to IDs
# =====================================================

# Every token is converted into a numerical value using the model's
# predefined vocabulary.

token_ids = tokenizer.convert_tokens_to_ids(tokens)

# print(token_ids)

# [1045, 1005, 1049, 2061, 7568, 2000,
#  2022, 4083, 2055, 2312, 2653, 4275, 1012]

# Each token now has its own numerical representation.


# =====================================================
# Decode the Token IDs
# =====================================================

# We can convert the token IDs back into text to verify the results.

decoded_ids = tokenizer.decode(token_ids)

# print(decoded_ids)

# i ' m so excited to be learning about large language models.


# =====================================================
# Special Tokens
# =====================================================

# The token IDs above are contained within the full input_ids output.
# However, input_ids also contains additional special tokens.

# Let's decode those values.

# print(tokenizer.decode(101))
# [CLS]

# print(tokenizer.decode(102))
# [SEP]

# These are special tokens automatically added by the tokenizer.

# Different models use different special tokens depending on their
# architecture.


# =====================================================
# XLNet Tokenization
# =====================================================

# Tokenization changes depending on which model we use.

# Every language model has its own vocabulary, tokenization strategy,
# and special tokens.

model2 = "xlnet-base-cased"

# Load the tokenizer for the XLNet model.

tokenizer2 = AutoTokenizer.from_pretrained(model2)

input_ids = tokenizer2(sentence)

# print(input_ids)

# {
#     'input_ids': [35, 26, 98, 102, 5564, 22, 39,
#                   1899, 75, 392, 1243, 2626, 9, 4, 3],
#     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0,
#                        0, 0, 0, 0, 0, 0, 2],
#     'attention_mask': [1, 1, 1, 1, 1, 1, 1,
#                        1, 1, 1, 1, 1, 1, 1, 1]
# }

# The numerical IDs are different from the BERT tokenizer, and the
# token_type_ids now contain more than just zeros.


# =====================================================
# Compare the Tokenization
# =====================================================

# Let's compare how XLNet tokenizes the same sentence.

tokens = tokenizer2.tokenize(sentence)

# print(tokens)

# ['▁I', "'", 'm', '▁so', '▁excited', '▁to', '▁be',
#  '▁learning', '▁about', '▁large', '▁language',
#  '▁models', '.']

# Notice that XLNet tokenizes the sentence differently than BERT.

token_ids = tokenizer2.convert_tokens_to_ids(tokens)

# print(token_ids)

# [35, 26, 98, 102, 5564, 22,
#  39, 1899, 75, 392, 1243, 2626, 9]

# The numerical IDs assigned to each token are also different.


# =====================================================
# XLNet Special Tokens
# =====================================================

# Unlike BERT, XLNet does not begin the sequence with a [CLS] token.

# Instead, the special tokens appear at the end of the sequence.

# print(tokenizer2.decode(4))
# <sep>

# print(tokenizer2.decode(3))
# <cls>

# This demonstrates why it is important to use the tokenizer that matches
# the language model. Each model has its own vocabulary, tokenization
# strategy, and special tokens.