# Ngrams.py

# N-grams help us analyze the relationship between neighboring words. An N-gram is simply a
# sequence of n tokens. The value of n tells us how many words are grouped together. When n
# equals one, we have single words called unigrams. When n equals two, we have pairs of
# consecutive words called bigrams. When n equals three, we have sequences of three words
# called trigrams.

# For example, in the sentence "I love NLP," the unigrams are "I", "love", and "NLP". The
# bigrams are "I love" and "love NLP". The trigram is "I love NLP". By working with N-grams,
# we can capture context that single words alone cannot. For instance, recognizing phrases
# like "New York" or "credit card", which carry a different meaning together than the words
# do separately.

import nltk
import pandas as pd
import matplotlib.pyplot as plt

# NLTK - The Natural Language Toolkit for tokenization, stemming, lemmatization,
# and many other text-processing tasks.

# pandas - A Python library for working with structured data. One of the most important
# Python libraries for data analysis. It is designed to make working with structured data
# fast and intuitive. The core feature in pandas is the DataFrame, which works like a
# spreadsheet inside Python. A DataFrame stores data in rows and columns and makes it easy
# to filter, sort, group, or transform the data. This makes pandas perfect for preparing
# text data for analysis, keeping it organized, and connecting it with other steps in our
# machine learning workflow.

# matplotlib - The most common Python library for creating charts and visualizations.
# With matplotlib, we can take the results of our text analysis and turn them into graphs,
# making it much easier to spot patterns and trends in the data.

tokens = ['the', 'rise', 'of', 'artificial', 'intelligence', 'has', 'led', 'to', 'significant', 'advancements', 'in', 'natural', 'language', 'processing', 'computer', 'vision', 'and', 'other', 'fields', 'machine', 'learning', 'algorithms', 'are', 'becoming', 'more', 'sophisticated', 'enabling', 'computers', 'to', 'perform', 'complex', 'tasks', 'that', 'were', 'once', 'thought', 'to', 'be', 'the', 'exclusive', 'domain', 'of', 'humans', 'with', 'the', 'advent', 'of', 'deep', 'learning', 'neural', 'networks', 'have', 'become', 'even', 'more', 'powerful', 'capable', 'of', 'processing', 'vast', 'amounts', 'of', 'data', 'and', 'learning', 'from', 'it', 'in', 'ways', 'that', 'were', 'not', 'possible', 'before', 'as', 'a', 'result', 'ai', 'is', 'increasingly', 'being', 'used', 'in', 'a', 'wide', 'range', 'of', 'industries', 'from', 'healthcare', 'to', 'finance', 'to', 'transportation', 'and', 'its', 'impact', 'is', 'only', 'set', 'to', 'grow', 'in', 'the', 'years', 'to', 'come']

print(tokens)

unigrams = (pd.Series(nltk.ngrams(tokens, 1))).value_counts()

# nltk.ngrams() takes two arguments: our list of tokens and the value of n.
# Since we want unigrams, we set n to one. This gives us a list of all the
# one-word sequences in our text.

# A pandas Series is like a single column of data in a spreadsheet. It lets us
# organize our unigrams into a structured format that pandas can work with.

# value_counts() goes through all of the unigrams and counts how many times each one appears.

bigrams = (pd.Series(nltk.ngrams(tokens, 2))).value_counts()

trigrams = (pd.Series(nltk.ngrams(tokens, 3))).value_counts()

print(trigrams)

# (to,)          7
# (of,)          6
# (the,)         4
# (in,)          4
# (learning,)    3
#               ..
# (humans,)      1
# (rise,)        1
# (advent,)      1
# (deep,)        1
# (come,)        1

unigrams[:10].sort_values().plot.barh(color="lightsalmon", width=.9, figsize=(12, 8))
plt.title("10 Most Frequently Occurring Unigrams")

# plt.show()  # Displays the bar chart.

# N = 1 => Unigrams
# N = 2 => Bigrams
# N = 3 => Trigrams
# N > 3 => N-grams