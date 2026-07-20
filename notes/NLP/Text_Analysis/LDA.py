import pandas as pd
import re  # imports Python's built-in re (Regular Expressions) module
from nltk.corpus import stopwords  # Used for removing stopwords
from nltk.tokenize import word_tokenize  # Used for splitting text into tokens
from nltk.stem import PorterStemmer  # A classic rule-based algorithm that reduces English words to a simpler base form
import gensim
import gensim.corpora  # Used to build the dictionary, corpus, and train the LDA model


# Topic modeling helps us uncover the main themes that run through a
# collection of documents.

# In this example, we'll use the Latent Dirichlet Allocation (LDA)
# algorithm to find topics in a collection of news articles.

# The overall process consists of three main steps:
# 1. Prepare the text.
# 2. Build the structures that LDA expects.
# 3. Train the model using Gensim.

# Gensim is a Python library designed for text analysis. It is lightweight,
# fast, and built to handle large collections of documents.

# We'll use Gensim to create the dictionary and corpus that LDA requires,
# and then train the model itself.

data = pd.read_csv('../../data/news_articles.csv')

# print(data.head())
# The content column holds the full text of each article, while the title
# column contains the article headlines.

# print(data.info())
# We confirm that the dataset contains 100 rows and no missing values,
# meaning every article is complete.

articles = data['content']  # Extract just the content column.

# Now we'll apply a series of preprocessing steps to the entire dataset by
# chaining together pandas apply() methods with lambda functions.

# Cleaning the text so it is ready for LDA. This includes:
# - Converting text to lowercase
# - Removing punctuation
# - Removing stopwords
# - Tokenizing
# - Stemming

# Convert every article to lowercase and remove punctuation.

# The lambda function takes one article at a time and applies re.sub() to
# remove punctuation. The apply() method automatically repeats this process
# for every article in the dataset.

articles = articles.str.lower().apply(lambda x: re.sub(r"([^\w\s])", "", x))

# Remove stopwords.

en_stopwords = stopwords.words("english")
articles = articles.apply(
    lambda x: " ".join([word for word in x.split() if word not in en_stopwords])
)

# Tokenize each article.

# Here, the lambda function applies word_tokenize(), which splits each
# article into a list of individual words (tokens).

articles = articles.apply(lambda x: word_tokenize(x))

# Stemming (chosen for speed because we have a large amount of text).

# Stemming is chosen instead of lemmatization because we have a large
# collection of documents, and stemming generally processes text faster.
# If preferred, lemmatization could also be used.

ps = PorterStemmer()

# Reduce every token to its stem.

articles = articles.apply(
    lambda tokens: [ps.stem(token) for token in tokens]
)

# print(articles)
# We can see that there are 100 items, each representing one cleaned and
# tokenized article.

# Create a dictionary that maps every unique word in our dataset to a
# unique integer ID.

dictionary = gensim.corpora.Dictionary(articles)

# print(dictionary)
# Each unique word receives its own ID, allowing the LDA model to work with
# the text in a structured format.

# We have 8,693 unique tokens, representing the number of unique words
# in our dataset.

# Dictionary<8693 unique tokens: ['10', '100', '108', '15', '155']...>

# Create the document-term matrix.

doc_term = [dictionary.doc2bow(text) for text in articles]

# print(doc_term)

# doc2bow stands for "document to Bag of Words." It takes a single article,
# looks up each word in the dictionary, and returns a list of word IDs along
# with the number of times each word appears.

# We use a list comprehension to apply doc2bow() to every article and store
# the results in the doc_term variable.

# Instead of plain text, each article is now represented as a Bag of Words
# vector, which is the format required by the LDA model.

# Before training the model, we decide how many topics we want it to find.

# For this example, we'll use two topics.

num_topics = 2

# Create the LDA model using Gensim.

# The model requires three main arguments:
# - corpus: the document-term matrix
# - id2word: the dictionary that maps IDs back to words
# - num_topics: the number of topics we want the model to discover

# Together, these inputs provide everything the model needs to identify
# recurring themes throughout the collection of articles.

lda_model = gensim.models.LdaModel(
    corpus=doc_term,
    id2word=dictionary,
    num_topics=num_topics
)

print(lda_model.print_topics(num_topics=num_topics, num_words=5))

# The first argument specifies how many topics to display.
# The second argument specifies how many words to display for each topic.

# [(0, '0.019*"mr" + 0.016*"said" + 0.007*"trump" + 0.005*"would" + 0.004*"state"'),
#  (1, '0.015*"mr" + 0.013*"said" + 0.004*"one" + 0.004*"state" + 0.004*"peopl"')]

# The output shows the most important words that the model has identified
# for each topic. These words help us understand what each topic is about.
#
# Since we are analyzing news articles, these topics are not especially
# informative. This suggests that we may need to adjust the number of topics,
# further clean the dataset, or explore the data more to produce more
# meaningful topics.