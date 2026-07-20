import pandas as pd
import re  # Used for cleaning the text
from nltk.corpus import stopwords  # Used for removing stopwords
from nltk.tokenize import word_tokenize  # Used for splitting text into tokens
from nltk.stem import PorterStemmer  # Reduces English words to a simpler base form
import gensim.corpora  # Used to build the dictionary and corpus
from gensim.models import LsiModel  # LSI and LSA are often used interchangeably in Gensim
from gensim.models.coherencemodel import CoherenceModel  # Used to calculate coherence scores
import matplotlib.pyplot as plt  # Used to compare coherence scores for different topic counts


# Topic modeling helps us uncover the main themes that run through a
# collection of documents.

# In this example, we'll use Latent Semantic Analysis (LSA) to find topics
# within a collection of news articles.

# Gensim refers to this model as Latent Semantic Indexing (LSI), so we use
# the LsiModel class. LSI and LSA are often used interchangeably.

# The overall process consists of four main steps:
# 1. Prepare the text.
# 2. Build the structures that LSA expects.
# 3. Compare models with different numbers of topics.
# 4. Train the final model using the selected number of topics.

# Gensim is a Python library designed for text analysis. It is lightweight,
# fast, and built to handle large collections of documents.

data = pd.read_csv('../../data/news_articles.csv')

# print(data.head())
# The content column holds the full text of each article, while the title
# column contains the article headlines.

# print(data.info())
# We confirm that the dataset contains 100 rows and no missing values,
# meaning every article is complete.

articles = data['content']  # Extract only the content column.

# Now we'll apply a series of preprocessing steps to the entire dataset by
# chaining together pandas methods with lambda functions.

# Cleaning the text so it is ready for LSA includes:
# - Converting text to lowercase
# - Removing punctuation
# - Removing stopwords
# - Tokenizing
# - Stemming

# Convert every article to lowercase and remove punctuation.

# The lambda function takes one article at a time and applies re.sub() to
# remove punctuation. The apply() method repeats this process for every
# article in the dataset.

articles = articles.str.lower().apply(
    lambda x: re.sub(r"([^\w\s])", "", x)
)

# Remove stopwords.

en_stopwords = stopwords.words("english")

articles = articles.apply(
    lambda x: " ".join(
        [word for word in x.split() if word not in en_stopwords]
    )
)

# Tokenize each article.

# The lambda function applies word_tokenize(), which splits each article
# into a list of individual words, or tokens.

articles = articles.apply(
    lambda x: word_tokenize(x)
)

# Stemming is chosen for speed because we have a large amount of text.

# Stemming reduces related word forms to the same stem. If preferred,
# lemmatization could also be used instead.

ps = PorterStemmer()

articles = articles.apply(
    lambda tokens: [ps.stem(token) for token in tokens]
)

# print(articles)
# We can see that there are 100 items, with each item representing one
# cleaned and tokenized article.

# Create a dictionary that maps every unique word in the dataset to a
# unique integer ID.

dictionary = gensim.corpora.Dictionary(articles)

# print(dictionary)
# Each unique word receives its own ID, allowing the LSA model to work
# with the text in a structured format.

# We have 8,693 unique tokens, representing the number of unique words
# in the dataset.

# Dictionary<8693 unique tokens: ['10', '100', '108', '15', '155']...>

# Create the document-term matrix.

doc_term = [
    dictionary.doc2bow(text)
    for text in articles
]

# print(doc_term)

# doc2bow stands for "document to Bag of Words." It takes a single article,
# looks up each word in the dictionary, and returns a list of word IDs along
# with the number of times each word appears.

# We use a list comprehension to apply doc2bow() to every article and store
# the results in the doc_term variable.

# Instead of plain text, each article is now represented as a Bag of Words
# vector, which is the format required by the LSA model.

# Before training the model, we choose an initial number of topics.

num_topics = 2

# Create an initial LSA model using Gensim's LsiModel class.

# The model requires three main arguments:
# - corpus: the document-term matrix
# - num_topics: the number of topics we want the model to discover
# - id2word: the dictionary that maps word IDs back to actual words

lsamodel = LsiModel(
    corpus=doc_term,
    num_topics=num_topics,
    id2word=dictionary
)

# print(lsamodel.print_topics(num_topics=num_topics, num_words=5))

# The first argument specifies how many topics to display.
# The second argument specifies how many words to display for each topic.

# [(0, '0.615*"mr" + 0.429*"said" + 0.187*"trump" + 0.130*"state" + 0.119*"would"'),
#  (1, '-0.537*"mr" + -0.319*"trump" + 0.286*"said" + 0.242*"saudi" + 0.142*"weight"')]

# A coherence score tells us how meaningful the top words in a topic are
# when grouped together. Higher coherence scores usually mean that the
# topics make more sense to humans.

# We'll create models using different numbers of topics and store their
# coherence scores in a list so that we can compare them.

# Create one empty list for the coherence scores and another for the models.

coherence_values = []
model_list = []

# Specify the minimum and maximum numbers of topics that we want to test.

min_topics = 2
maximum_topics = 11

# Create a loop that tests every number of topics from the minimum to the
# maximum.

for num_topics_i in range(min_topics, maximum_topics + 1):

    # Create a new LSA model during each iteration.
    #
    # The number of topics is controlled by num_topics_i. Each time the loop
    # runs, num_topics_i increases by one until it reaches maximum_topics.
    #
    # random_seed=0 helps produce consistent results each time the code runs.

    model = LsiModel(
        corpus=doc_term,
        num_topics=num_topics_i,
        id2word=dictionary,
        random_seed=0
    )

    model_list.append(model)

    # CoherenceModel receives:
    # - The model we just created
    # - The original tokenized articles
    # - The dictionary
    # - The coherence method
    #
    # The c_v coherence method measures how often the most important words
    # within a topic appear together in the documents. It is widely used
    # because its results often align with human judgments of topic quality.

    coherence_model = CoherenceModel(
        model=model,
        texts=articles,
        dictionary=dictionary,
        coherence='c_v'
    )

    # Calculate the coherence score and add it to coherence_values.

    coherence_values.append(
        coherence_model.get_coherence()
    )

# Plot the coherence scores for each number of topics.

plt.plot(
    range(min_topics, maximum_topics + 1),
    coherence_values
)

plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.legend(["Coherence Values"], loc='best')
plt.tight_layout()

# plt.show()

# In our case, the coherence score reaches its highest point at three topics.
# This suggests that a model with three topics produces the most meaningful
# grouping of words.

# Create the final model using three topics.

final_n_topics = 3

lsamodel_final = LsiModel(
    corpus=doc_term,
    num_topics=final_n_topics,
    id2word=dictionary,
    random_seed=0
)

print(
    lsamodel_final.print_topics(
        num_topics=final_n_topics,
        num_words=5
    )
)

# [(0, '0.615*"mr" + 0.429*"said" + 0.187*"trump" + 0.130*"state" + 0.119*"would"'),
#  (1, '-0.537*"mr" + -0.319*"trump" + 0.286*"said" + 0.242*"saudi" + 0.142*"weight"'),
#  (2, '0.460*"saudi" + 0.264*"taliban" + -0.249*"weight" + 0.194*"afghanistan" + -0.191*"dr"')]