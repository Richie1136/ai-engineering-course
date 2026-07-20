# TextPreprocessing.py

# The Importance of Data Preparation

# In natural language processing (NLP), one of the most important factors that will determine the
# accuracy of any machine learning or insights that you're trying to get is the quality of data that you
# provide and how that data has been cleaned up. If you're feeding an algorithm garbage data, meaning data that
# has not been cleaned up properly, is not in the right format, and has loads of noise in it, then the
# accuracy of your machine learning is going to be affected by this.

# There are a number of steps involved in preprocessing our text data and getting it ready for further analysis.

# 1. General Cleaning - Taking our data set, getting it organized, tidying up the text, and removing anything that might throw an error.
# 2. Removing Noise from the Data Set - Removing aspects of the data that are not adding any value.
# 3. Getting the Data in the Right Format - Formatting the data for the ML algorithm.

# Preprocessing our text data will take it from looking something like this to this:

# "The quick brown fox jumps over the lazy dog" => "[quick, brown, fox, jump, lazy, dog]"

# Stemming: Reducing words to their base form. For example, words like connecting or connected
# will be reduced to the base form connect.

# Stemming works by chopping off endings or suffixes from words. The downside is that sometimes
# the result isn't a proper word or doesn't look meaningful. For instance, studies might be stemmed
# to something like stud. We standardize the text this way because it reduces the number of unique
# words in our data set. Using fewer unique words reduces complexity and keeps the data set smaller
# and easier to manage. By removing this extra noise, we make the data cleaner and simpler to process,
# which is an essential step in preparing it properly for machine learning.

from nltk.stem import PorterStemmer

# The Porter stemmer is a classic rule-based algorithm that reduces English words to a simpler base.

ps = PorterStemmer()

connect_tokens = ['connecting', 'connected', 'connectivity', 'connect', 'connects']

for t in connect_tokens:
    print(t, ": ", ps.stem(t))

# The code takes each word, stems it, and prints both the original word and its stem.

# connecting: connect
# connected: connect
# connectivity: connect
# connect: connect
# connects: connect


learn_tokens = ['learned', 'learning', 'learn', 'learns', 'learner', 'learners']

for t in learn_tokens:
    print(t, ": ", ps.stem(t))

# The code takes each word, stems it, and prints both the original word and its stem.

# learned: learn
# learning: learn
# learn: learn
# learns: learn
# learner: learner
# learners: learner


likes_tokens = ['likes', 'better', 'worse']

for t in likes_tokens:
    print(t, ": ", ps.stem(t))

# The code takes each word, stems it, and prints both the original word and its stem.

# likes: like
# better: better
# worse: wors  # <= Drawback because the output isn't meaningful.


# Lemmatization: Reduces a word to a meaningful base form while preserving its intended meaning.

# Lemmatization is more sophisticated because it references a predefined dictionary to find the correct
# base form of a word. Unlike stemming, it attempts to return real, meaningful words instead of simply
# removing prefixes or suffixes. The trade-off, however, is that we may still have more unique words in
# our dataset compared to stemming.

import nltk

nltk.download('wordnet')  # WordNet is an extensive lexical database of English.
# Basically, a built-in dictionary that the lemmatizer uses to make sure the base forms
# it produces are real words.

from nltk.stem import WordNetLemmatizer

w = WordNetLemmatizer()

for t in connect_tokens:
    print(t, ": ", w.lemmatize(t))

# w.lemmatize(t) takes the token t and looks it up in the WordNet dictionary to return its proper form.

# connecting: connecting
# connected: connected
# connectivity: connectivity
# connect: connect
# connects: connects


for t in learn_tokens:
    print(t, ": ", w.lemmatize(t))

# learned: learned
# learning: learning
# learn: learn
# learns: learns
# learner: learner
# learners: learner


for t in likes_tokens:
    print(t, ": ", w.lemmatize(t))

# likes: like
# better: better
# worse: worse