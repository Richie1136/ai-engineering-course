# Tokenization.py

# One of the most critical steps in natural language processing (NLP)
# is breaking text into smaller units. This process is called tokenization,
# and the smaller units are known as tokens.

# The most common type of tokenization is word tokenization, where each word in a sentence
# becomes a token. However, tokens don't have to be words.

# Depending on your use case, they could also be sentences, subwords, or even single characters.

# We use tokenization because breaking text into smaller parts makes it easier to analyze and understand.

# For example, looking at individual words can reveal patterns, such as which words are most common
# or how often specific terms appear together.

import nltk

nltk.download('punkt_tab')  # A table used to determine where sentences begin and end.

from nltk.tokenize import word_tokenize, sent_tokenize

# word_tokenize - Splits text into words.
# sent_tokenize - Splits text into sentences.

sentence = "Her cat's name is Luna. Her dog's name is Max."

# Place the sentence variable inside sent_tokenize() to split the string into individual sentences.
print(sent_tokenize(sentence))

sentence = "Her cat's name is Luna"

# The sentence is split into individual tokens, where each word becomes its own token.
print(word_tokenize(sentence))

sentence_2 = "Her cat's name is Luna and her dog's name is Max"

print(word_tokenize(sentence_2))

# Capitalized and lowercase versions of the same word might be treated as separate entries.

# By converting everything to lowercase, we can ensure consistency.