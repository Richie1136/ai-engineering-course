"""Fake-news NLP case study.

This script walks through a complete NLP workflow:
1. Explore fake and factual news articles.
2. Compare part-of-speech tags and named entities.
3. Clean and preprocess article text.
4. Analyze sentiment with VADER.
5. Discover topics with LDA and LSA.
6. Train classifiers to distinguish fake from factual news.
"""

# =============================================================================
# Imports
# =============================================================================

import re

import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import spacy
from gensim.models import LsiModel, TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =============================================================================
# Project Overview
# =============================================================================

# Going to be running through a practical example, taking one data set and running through all the steps that we've covered in this course in a real life
# business setting.


# Imagine you're working for a social media company, and the company is concerned with the growing amount of fake news circulating on its platform.


# They've assigned you, as a data scientist, to investigate how fake news can be recognized, and create a method of identifying it.


# First step is to explore and clean the data, and then working to classify fake versus factual news stories.

# We'll also create some plots of our outputs and discuss how we will communicate our findings to stakeholders.

# Using pandas package for data manipulation, matplotlib and seaborn for plotting, spacy, re, NLTK, gensim and sklearn for all different types of analysis that we want to do.

# Going to set some plot options to get started.

# Going to set the figure size as 12.8 to make sure all of our charts are printed in a nice size, and we'll also specify a default plot color to use.


# =============================================================================
# Plot Configuration
# =============================================================================

# Set plot options
plt.rcParams['figure.figsize'] = (12,8)
default_plot_color = '#00bfbf'


# =============================================================================
# Load Dataset
# =============================================================================

data = pd.read_csv('../../data/fake_news_data.csv')
# print(data.head())
# print(data.info())

# This is the response for data.info()

# <class 'pandas.core.frame.DataFrame'>
#RangeIndex: 198 entries, 0 to 197
#Data columns (total 4 columns):
 #   Column           Non-Null Count  Dtype 
#---  ------           --------------  ----- 
# 0   title            198 non-null    object
# 1   text             198 non-null    object
# 2   date             198 non-null    object
# 3   fake_or_factual  198 non-null    object
#dtypes: object(4)
#memory usage: 6.3+ KB
#None



# data['fake_or_factual'].value_counts().plot(kind='bar', color=default_plot_color) # Plotting a bar chart for the fake_or_factual column
# plt.title('Count of Article Classification')
# plt.show() # We have a similar number of factual and fake articles but if we did have a significantly different numbers of rows in each groups, we would need
# to take some steps to make this a balanced data set.


# =============================================================================
# Part-of-Speech (POS) Tagging
# =============================================================================

# Because one of our tasks is to determine the differences between fake and factual news, we want to split out the dataset into fake news and factual news.

# And then we can compare the POS tags that occur between each of the different datasets.

nlp = spacy.load('en_core_web_sm')


# Creating our fake news dataset, just taking our dataframe and splitting it out where the column fake_or_factual equals fake news or factual news
fake_news = data[data['fake_or_factual'] == "Fake News"]
fact_news = data[data['fake_or_factual'] == 'Factual News']

# Creating our two separate spacy documents

# First we'll take our fake dataset and create fake_spacydocs.

# Because we're working over a data frame, we'll want to use nlp.pipe over our fake news text column and do the same for fact news

fake_spacydocs = list(nlp.pipe(fake_news['text']))
fact_spacydocs = list(nlp.pipe(fact_news['text']))


# Create function to extract the tags for each of the documents in our data, so each of the rows in our data frame.

def extract_token_tags(doc:spacy.tokens.doc.Doc):
    # Will return text the ent type and the pos tags for each item in the document.
    return [(i.text, i.ent_type_, i.pos_) for i in doc] # We're not only extracting the POS tags, but the entity type for named entity recognition


# Tagging our data set

# Taking the fake data set and creating an empty list

fake_tagdf = []

# Specifying the column names we want to use

columns = ["token", "ner_tag", "pos_tag"] # ner_tag for the named entities

# Iterating through each of the documents. So each of the rows in our dataframe and pull out the relevant tags.


# Creating for loop

for ix, doc in enumerate(fake_spacydocs): # Taking the tags by using our extract token tags function
    tags = extract_token_tags(doc)
    tags = pd.DataFrame(tags) # Converting these tags into a dataframe.
    tags.columns = columns # give the column names as the columns we specified earlier
    fake_tagdf.append(tags)


# Once we've run through each of the documents and we've populated our fake_tagedf, we'll then use pd.concat to get it into the right function and assign this back to fake tags.

fake_tagdf = pd.concat(fake_tagdf)

fact_tagdf = []

for ix,doc in enumerate(fact_spacydocs):
    tags = extract_token_tags(doc)
    tags = pd.DataFrame(tags)
    tags.columns = columns
    fact_tagdf.append(tags)

fact_tagdf = pd.concat(fact_tagdf)

# print(fake_tagdf.head())

# Response from fake_tagdf.head()

# As we can see here this data set has been broken out into the individual tokens, named entities have been pulled out where appropriate and
# each token has been given its relevant POS tag.

#       token   ner_tag pos_tag
# 0     There              PRON
# 1       are              VERB
# 2       two  CARDINAL     NUM
# 3     small               ADJ
# 4  problems              NOUN


# Take token frequency count

pos_counts_fake = fake_tagdf.groupby(['token', 'pos_tag']).size().reset_index(name="counts").sort_values(by="counts", ascending=False) # Grouping by the token and the pos_tag column
# print(pos_counts_fake.head(10))

# Response from pos_counts_fake.head(10)

# Because we havent cleaned the data set yet, you can see that this is just punctuation and stop words. This helps when you come to removing the stop words you can reference this list and make sure
# that any kind of really frequently occurring words that you don't want in your data set are properly included within the stop words.

#      token pos_tag  counts
# 29       ,   PUNCT    1908
# 7451   the     DET    1834
# 41       .   PUNCT    1530
# 5766    of     ADP     922
# 2665   and   CCONJ     875
# 2449     a     DET     805
# 0            SPACE     795
# 7528    to    PART     767
# 4921    in     ADP     668
# 5099    is     AUX     419



pos_counts_fact = fact_tagdf.groupby(['token', 'pos_tag']).size().reset_index(name="counts").sort_values(by="counts", ascending=False) # Grouping by the token and the pos_tag column
# print(pos_counts_fact.head(10))

# Response from pos_counts_fact.head(10)

# 6145   the     DET    1903
# 14       ,   PUNCT    1698
# 21       .   PUNCT    1382
# 4716    of     ADP     884
# 1898     a     DET     789
# 2093   and   CCONJ     757
# 4005    in     ADP     671
# 6205    to    PART     660
# 4743    on     ADP     482
# 5567  said    VERB     451


# Look at the frequencies of individual POS tags. So how many nouns occur?

# Will be interesting to compare this between the fake and factual data set to see if different types of tags are coming up more frequently.


# print(pos_counts_fake.groupby('pos_tag')['token'].count().sort_values(ascending=False).head(10))

# Response from pos_counts_fake.groupby('pos_tag')['token'].count().sort_values(ascending=False).head(10)

# pos_tag
# NOUN     2586
# VERB     1817
# PROPN    1672
# ADJ       882
# ADV       413
# NUM       221
# PRON       96
# ADP        89
# AUX        62
# SCONJ      51
# Name: token, dtype: int64


# print(pos_counts_fact.groupby('pos_tag')['token'].count().sort_values(ascending=False).head(10))

# Response from pos_counts_fact.groupby('pos_tag')['token'].count().sort_values(ascending=False).head(10)


# NOUN     2179
# VERB     1539
# PROPN    1379
# ADJ       747
# ADV       263
# NUM       205
# PRON       79
# ADP        70
# AUX        43
# SCONJ      42
# Name: token, dtype: int64


# Deeper dive into the nouns used by each of the data sets.

# print(pos_counts_fake[pos_counts_fake.pos_tag == 'NOUN'][:15]) # Taking our data set, filtering it to where the pos_tag equals NOUN and then giving the top 15 results

# Response from pos_counts_fake[pos_counts_fake.pos_tag == 'NOUN'][:15]

#            token pos_tag  counts
# 5981      people    NOUN      77
# 7349           t    NOUN      65
# 6216   president    NOUN      58
# 7960       women    NOUN      55
# 7516        time    NOUN      52
# 3138    campaign    NOUN      44
# 8011        year    NOUN      44
# 4581  government    NOUN      41
# 5213         law    NOUN      40
# 8013       years    NOUN      40
# 7165       state    NOUN      38
# 4012    election    NOUN      37
# 3643         day    NOUN      35
# 5482       media    NOUN      35
# 3538     country    NOUN      33



# print(pos_counts_fact[pos_counts_fact.pos_tag == 'NOUN'][:15])

# Response from pos_counts_fact[pos_counts_fact.pos_tag == 'NOUN'][:15]


#                token pos_tag  counts
# 3738      government    NOUN      71
# 6618            year    NOUN      64
# 5901           state    NOUN      57
# 2360            bill    NOUN      55
# 1975  administration    NOUN      51
# 5066       president    NOUN      49
# 3277        election    NOUN      48
# 4919          people    NOUN      45
# 4786           order    NOUN      45
# 4259             law    NOUN      42
# 2497        campaign    NOUN      42
# 6095             tax    NOUN      39
# 5396       reporters    NOUN      38
# 5905       statement    NOUN      37
# 2873           court    NOUN      37



# =============================================================================
# Named Entity Recognition (NER)
# =============================================================================

# It is better to be look at the named entities before you do any kind of text pre-processing.

# We are doing it now, to give our models the best chance of pulling out those interesting named entities before we go in and do any kind of 
# preprocessing and cleaning of the data. Because we already pulled out the named entity tags in our last lesson when we did POS tagging, we don't have to do that step again,
# We can just jump straight in and start looking at what's been pulled out.


# Want to start by pulling out the top entities in our fake news data set.

# This is our fake data set filtered to where the ner_tage column is not blank.
# So this means going through and taking any token that has been recognized to have a named entity in there.
# We then want to group by the token column and the ner_tag column.
# Take the size, reset the index name it counts, and then sort our values by these counts in descending order
top_entities_fake = fake_tagdf[fake_tagdf['ner_tag'] != ''].groupby(['token', 'ner_tag']).size().reset_index(name='counts').sort_values('counts',ascending=False)

top_entities_fact = fact_tagdf[fact_tagdf['ner_tag'] != ''].groupby(['token', 'ner_tag']).size().reset_index(name='counts').sort_values('counts',ascending=False)

# print(top_entities_fake)

# print(top_entities_fact)


# Creating some nice plots of the named entities that have been pulled out, so that we can use this in presentations or take it to stakeholders.


# Creating a color palette to make sure that both plots are in the same colors for the different entity tags.

# So for example, if we've got different people being pulled out of the different data sets, we want to make sure that it's going to be represented by the same
# color in both of our plots. So it looks nice and is easy to understand

ner_palette = {
    'ORG': sns.color_palette("Set2").as_hex()[0],
    'GPE': sns.color_palette("Set2").as_hex()[1],
    'NORP': sns.color_palette("Set2").as_hex()[2],
    'PERSON': sns.color_palette("Set2").as_hex()[3],
    'DATE': sns.color_palette("Set2").as_hex()[4],
    'CARDINAL': sns.color_palette("Set2").as_hex()[5],
    'PERCENT': sns.color_palette("Set2").as_hex()[6],
}


# Using the Seaborn package to create some really nice looking plots

# sns.barplot(
#     # x is going to be our counts
#     # y is going to be the token
#     # hue which is the color to use for each bar
#     x = "counts",
#     y = "token",
#     hue = 'ner_tag', # Wanting the color by the ner_tag
#     palette=ner_palette,
#     data = top_entities_fake[:10],
#     orient='h', # Horizontal bar chart
#     dodge=False # Get the bar is the right format
# ).set(title="Most Common Entities in Fake News")

# plt.show()

# The most common named entities that have been pulled out are people

# Fact News

# sns.barplot(
#     # x is going to be our counts
#     # y is going to be the token
#     # hue which is the color to use for each bar
#     x = "counts",
#     y = "token",
#     hue = 'ner_tag', # Wanting the color by the ner_tag
#     palette=ner_palette,
#     data = top_entities_fact[:10],
#     orient='h', # Horizontal bar chart
#     dodge=False # Get the bar is the right format
# ).set(title="Most Common Entities in Fact News")

# plt.show() 

# Here people's names are much less common. Here organizations and place names seem to be much more common, and there's less of a focus on people.


# =============================================================================
# Text Preprocessing
# =============================================================================

# So far, we've loaded in our data set and explored it a little using parts of speech tagging and named entity recognition.


# print(data.head()) # title column, text column, the date, and the fake or factual tags

#                                                title                                               text                 date fake_or_factual
#0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News
#1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News
#2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News
#3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News
#4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News


# Creating regular expression to look for the hyphen in the text column and remove everything before the first hyphen


# Creating new column that is going to contain all of our cleaned up text.

# Using data.apply to apply our re.sub function over the rows in our data set
# In our regular expression we are looking for the first hyphen in the text and remove the hyphen and everything before it.
# We specify the regular expression syntax and we want to replace this with blank, so essentially remove it from our text.
# Then we specify we want to run this over the text column and our axis = 1
data['text_clean'] = data.apply(lambda x: re.sub(r"^[^-]*-\s", "", x['text']), axis=1)

# print(data.head())

# Our text clean column has been added and the location tags have been removed.

#                                                title                                               text                 date fake_or_factual                                         text_clean
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News  There are two small problems with your analogy...
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News  Buried in Trump s bonkers interview with New Y...
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News  Women make up over 50 percent of this country,...
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News  U.S. Defense Secretary Jim Mattis said on Mond...
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News  Northern Ireland s political parties are rapid...



# -----------------------------
# Convert Text to Lowercase
# -----------------------------

data['text_clean'] = data['text_clean'].str.lower()

# -----------------------------
# Remove Punctuation
# -----------------------------

# Inside the regex looking for anything that isn't a word or a space and removing it

data['text_clean'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['text_clean']), axis=1)

# print(data.head())

#                                                title                                               text                 date fake_or_factual                                         text_clean
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News  there are two small problems with your analogy...
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News  buried in trump s bonkers interview with new y...
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News  women make up over 50 percent of this country ...
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News  us defense secretary jim mattis said on monday...
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News  northern ireland s political parties are rapid...


# -----------------------------
# Remove Stop Words
# -----------------------------

# When we did POS tagging above we had really common tokens that were mostly stopwords, so its really important to go back and sync that up with your
# stop words list and make sure that all those really common occurring words that you can remove are in that stop words list to be removed.

en_stopwords = stopwords.words('english')

data['text_clean'] = data['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

# print(data.head())

#                                                title                                               text                 date fake_or_factual                                         text_clean
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News  two small problems analogy susan jesus muslim ...
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News  buried trump bonkers interview new york times ...
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News  women make 50 percent country grossly underrep...
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News  us defense secretary jim mattis said monday ru...
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News  northern ireland political parties rapidly run...



# -----------------------------
# Tokenization
# -----------------------------

# This will go through each of our rows in our data set
# Take the text from text_clean and converts it into our word tokens.

data['text_clean'] = data.apply(lambda x: word_tokenize(x['text_clean']), axis=1)


# -----------------------------
# Lemmatization
# -----------------------------

# Going to be using Lemmatizing instead of stemming, because i want to use this more intelligent method to keep a lot of the context
# and meaning of the words where possible
# 

lemmatizer = WordNetLemmatizer()

# We use lemmatizer.lemmatize and in the brackets specify our tokens. for token in tokens

data['text_clean'] = data['text_clean'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

# print(data.head())


#                   title                                               text                 date fake_or_factual                                         text_clean
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News  [two, small, problem, analogy, susan, jesus, m...
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News  [buried, trump, bonkers, interview, new, york,...
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News  [woman, make, 50, percent, country, grossly, u...
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News  [u, defense, secretary, jim, mattis, said, mon...
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News  [northern, ireland, political, party, rapidly,...


# Creating a list of our clean tokens and then go on to look at the Unigrams

# =============================================================================
# N-Gram Analysis
# =============================================================================

# -----------------------------
# Unigrams
# -----------------------------
tokens_clean = sum(data['text_clean'], [])

# Using nltk.ngrams over our tokens_clean, we specify the value of n as one because we're just interested in unigrams.

unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()).reset_index()[:10]

# Printing out to get our top ten most frequently occurring unigrams.

# print(unigrams)

# This is much more interesting after we've done our pre-processing. It is not just stopwords anymore that are coming up as our
# most frequent tokens. We've got much more interesting words coming out here.

# (said,)          580
# (trump,)         580
# (u,)             277
# (state,)         275
# (president,)     259
# (would,)         226
# (one,)           160
# (clinton,)       141
# (year,)          139
# (republican,)    137
# Name: count, dtype: int64

# Turning this into a plot

unigrams['token'] = unigrams['index'].apply(lambda x: x[0])

# sns.barplot(x="count", y="token", data=unigrams, orient="h", palette=[default_plot_color], hue="token", legend=False).set(title="Most Common Unigrams After Preprocessing")

# plt.show()


# -----------------------------
# Bigrams
# -----------------------------

bigrams = (pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()).reset_index()[:10]

# Printing out to get our top ten most frequently occurring unigrams.

# print(bigrams)

# Got some interesting bigrams coming out such as united state, white house

#                      index  count
# 0          (donald, trump)    113
# 1          (united, state)     84
# 2           (white, house)     74
# 3      (president, donald)     47
# 4       (hillary, clinton)     39
# 5              (new, york)     33
# 6         (supreme, court)     30
# 7             (image, via)     29
# 8         (official, said)     26
# 9  (trump, administration)     26


# =============================================================================
# Sentiment Analysis (VADER)
# =============================================================================

# Trying to answer the question does sentiment different between the different news types. For example, does factual news have a more positive or more negative skewed sentiment
# as compared to the fake news dataset?


# Using Vader Sentiment calculation to get our sentiment analysis
vader_sentiment = SentimentIntensityAnalyzer()

# Creating column for our sentiment score

# Using the raw data to calculate this
# Applying the lambda function and take our vader sentiment polarity score from our text
data['vader_sentiment_score'] = data['text'].apply(lambda x: vader_sentiment.polarity_scores(x)['compound'])
# print(data.head())

# We can see our Vader Sentiment score with the relevant sentiment score for each of our documents

#                               title                                               text                 date fake_or_factual                                         text_clean  vader_sentiment_score
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News  [two, small, problem, analogy, susan, jesus, m...                -0.3660
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News  [buried, trump, bonkers, interview, new, york,...                -0.8197
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News  [woman, make, 50, percent, country, grossly, u...                 0.9779
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News  [u, defense, secretary, jim, mattis, said, mon...                -0.3400
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News  [northern, ireland, political, party, rapidly,...                 0.8590


# Classify into positive, negative and neutral sentiment

# -1 => -0.1 is our negative classification
# 0.1 => 1 is our positive classification
# -0.1 => 0.1 is our neutral classification
bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']


# pd.cut takes our vader_sentiment_score column, chops it up by the different bins, and gives them the labels that we specified in our names above
data['vader_sentiment_label'] = pd.cut(
    data['vader_sentiment_score'],
    bins=bins,
    labels=names,
    include_lowest=True,
)

# print(data.head())

# You can see that all the sentiment scores that are less than -0.1 have been classified as negative and there are a few positives as well 
# that have been correctly categorized.

#                                                title                                               text                 date  ...                                         text_clean vader_sentiment_score  vader_sentiment_label
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015  ...  [two, small, problem, analogy, susan, jesus, m...               -0.3660               negative
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017  ...  [buried, trump, bonkers, interview, new, york,...               -0.8197               negative
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016  ...  [woman, make, 50, percent, country, grossly, u...                0.9779               positive
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017   ...  [u, defense, secretary, jim, mattis, said, mon...               -0.3400               negative
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017   ...  [northern, ireland, political, party, rapidly,...                0.8590               positive

# [5 rows x 7 columns]


# Creating a bar chart of the different labels in each of the data.

# First going to look overall, how many positive, negative, and neutral sentiments we have across the whole data set. 

# data['vader_sentiment_label'].value_counts().plot.bar(color=default_plot_color)

# Looking at the chart, overall we've got a few more news articles with a positive sentiment as opposed to a negative sentiment, and
# we've got a few neutrals in there. Most of them have been classified as positive or negative.

# plt.show()

# Checking to see how this varies between the two types of news articles

# sns.countplot(
#     x = "fake_or_factual",
#     hue = "vader_sentiment_label",
#     palette=sns.color_palette("hls"),
#     data=data
# ).set(title="Sentiment by News Type")

# When looking at the fake news there's a pretty even split between positive and negative sentiment and a few neutrals but there isn't really a massive
# skew to one or the other. In our factual news dataset you can see that positive does skew more in this dataset. So we have a few more positive articles as 
# opposed to negative in the factual news dataset, there's also a lot less neutrals in this.

# plt.show()

# =============================================================================
# Topic Modeling with LDA
# =============================================================================

# We want to convert our text into numbers that can be fed into the LDA algorithm

# This is going to leave us with the cleaned up text from any article that's been classified as fake news
fake_news_text = data[data['fake_or_factual'] == 'Fake News']['text_clean'].reset_index(drop=True)

# Creating dictionary containing every unique token and assigns each token a numerical ID

dictionary_fake = corpora.Dictionary(fake_news_text)

# Create our bag of words and loops through every article and converts it into (token ID, count) pairs

doc_term_fake = [dictionary_fake.doc2bow(text) for text in fake_news_text]

# Generating coherent scores to determine the optimal number of topics

# Empty list for our coherence values and create an empty list for our models


min_topics = 2
max_topics = 11

def calculate_coherence_values():
    coherence_values = []
    model_list = []

    for num_topics_i in range(min_topics, max_topics + 1):
        model = gensim.models.LdaModel(
            doc_term_fake,
            num_topics=num_topics_i,
            id2word=dictionary_fake
        )

        model_list.append(model)



        coherence_model = CoherenceModel(
            model=model,
            texts=fake_news_text,
            dictionary=dictionary_fake,
            coherence="c_v"
        )

        coherence_values.append(coherence_model.get_coherence())

    # plt.plot(range(min_topics, max_topics + 1), coherence_values)
    # plt.xlabel("Number of Topics")
    # plt.ylabel("Coherence Score")
    # plt.legend(("coherence_values"), loc="best")
    # plt.show()

    return model_list, coherence_values


# Creating a for loop to run over each of the iterations of the model.

# for num_topics_i in range(min_topics, max_topics + 1):
#     # Train an LDA model on the Bag-of-Words articles to discover a specified number of topics
#     model = gensim.models.LdaModel(doc_term_fake, num_topics=num_topics_i, id2word=dictionary_fake)
#     # Appending model to model_list
#     model_list.append(model)
#     # Evaluate how semantically related the words within the model's topics are
#     coherence_model = CoherenceModel(model=model, texts=fake_news_text, dictionary=dictionary_fake, coherence="c_v")
#     # Calculate and store the current model's coherence score
#     coherence_values.append(coherence_model.get_coherence())

    # This loop will leave us with all our models in model list and a list of the coherence values put into our coherence values list.

# Plotting the coherence_values to see what they look like for each iteration of our model, using a different number of topics.


# Coherence score was highest when the number of topics was set to 9

# Two topics might not give us enough interest, whereas 11 topics is going to be too difficult to explain to different stakeholders and get an understanding 
# of what those topics are around.


# Although nine topics produced the highest coherence score, seven topics are used
# here to keep the final model easier to interpret and explain to stakeholders.

num_topics_lda = 7
lda_model = gensim.models.LdaModel(corpus=doc_term_fake, id2word=dictionary_fake, num_topics=num_topics_lda) # Creating LDA model


# Using lda_model.print_topics to print out the words that occur with each topic. We want to see the ten most common words for each of the topics
lda_model.print_topics(num_topics_lda, num_words=10)
# print(lda_model.print_topics(num_topics_lda, num_words=10))

# When we look through these topics, we can see that there is quite a lot of overlap, for example, the word president is a really frequently
# occurring word in all of the topics, as is the word said and other words throughout there as well also appears in a number of different topics

# [(0, '0.016*"trump" + 0.004*"said" + 0.004*"state" + 0.004*"president" + 0.004*"clinton" + 0.004*"u" + 0.003*"woman" + 0.003*"would" + 
# 0.003*"donald" + 0.003*"one"'), (1, '0.016*"trump" + 0.007*"president" + 0.005*"state" + 0.004*"said" + 0.004*"time" + 0.003*"woman" + 
# 0.003*"clinton" + 0.003*"obama" + 0.003*"u" + 0.003*"one"'), (2, '0.008*"trump" + 0.006*"said" + 0.005*"clinton" + 0.004*"president" + 
# 0.004*"state" + 0.004*"republican" + 0.004*"would" + 0.004*"hillary" + 0.003*"u" + 0.003*"time"'), (3, '0.010*"trump" + 0.005*"clinton" + 
# 0.004*"u" + 0.004*"email" + 0.003*"mccain" + 0.003*"would" + 0.003*"said" + 0.003*"state" + 0.003*"official" + 0.003*"law"'), 
# (4, '0.007*"said" + 0.006*"trump" + 0.004*"state" + 0.003*"mccain" + 0.003*"would" + 0.003*"party" + 0.003*"u" + 0.003*"president" + 
# 0.003*"republican" + 0.003*"clinton"'), (5, '0.004*"trump" + 0.003*"state" + 0.003*"one" + 0.003*"obama" + 0.003*"would" + 0.003*"said" + 
# 0.003*"republican" + 0.003*"u" + 0.003*"year" + 0.003*"people"'), (6, '0.011*"trump" + 0.005*"one" + 0.005*"president" + 0.004*"clinton" + 
# 0.004*"would" + 0.004*"said" + 0.004*"u" + 0.003*"republican" + 0.003*"donald" + 0.003*"year"')]


# Using the TF-IDF vectorization of our text, as opposed to just the bag of words model. We are going to be using latent semantic analysis as opposed to
# latent Dirichlet allocation, to see if that gives us different results

# =============================================================================
# Topic Modeling with LSA
# =============================================================================

# -----------------------------
# Create TF-IDF Corpus
# -----------------------------

# Define a function to create our TF-IDF corpus

def tfidf_corpus(doc_term_matrix):
    # Creating Tfidf Model
    # normalize = True scales each document’s TF-IDF vector so documents of different lengths can be compared more fairly
    tfidf = TfidfModel(corpus=doc_term_matrix, normalize=True)
    corpus_tfidf = tfidf[doc_term_matrix]
    return corpus_tfidf

# Create function to get the coherence scores, so we can find the most optimal value for a number of topics

# The arguments it wants to receive are the corpus, the dictionary, the text, the minimum number of topics, and the maximum number of topics we want to test

def get_coherence_scores(corpus, dictionary, text, min_topics, max_topics):
    coherence_values = []
    model_list = []
    for num_topics_i in range(min_topics, max_topics + 1):
        model = LsiModel(corpus, num_topics=num_topics_i, id2word=dictionary)
        model_list.append(model)
        coherence_model = CoherenceModel(model, texts=text, dictionary=dictionary, coherence="c_v")
        coherence_values.append(coherence_model.get_coherence())
# The plot will be plotting the number of topics against the coherence values
    # plt.plot(range(min_topics, max_topics + 1), coherence_values)
    # plt.xlabel("Number of Topics")
    # plt.ylabel("Coherence Score")
    # plt.legend(("coherence_value"), loc="best")
    # plt.show()

    return model_list, coherence_values

# Looks like the optimal number of topics is 4

# Creating our tf-idf representation of the text

# Take my fake-news articles represented with Bag-of-Words, convert them into TF-IDF representations, 
# and save the result as corpus_tfidf_fake.



if __name__ == "__main__":
    model_list, coherence_values = calculate_coherence_values()

    corpus_tfidf_fake = tfidf_corpus(doc_term_fake)

    lsa_models, lsa_coherence_values = get_coherence_scores(
        corpus_tfidf_fake,
        dictionary_fake,
        fake_news_text,
        min_topics=2,
        max_topics=11,
    )

    lsa_model = LsiModel(corpus_tfidf_fake, id2word=dictionary_fake, num_topics=7)
    lsa_model.print_topics()
    print(lsa_model.print_topics())

    # These are much more interesting topics to discuss and dig into what these differences are around the different topics and what they might mean

    # So our first topic seems to have the words Trump, Clinton and Hillary
    # The second topic boiler, acr, room
    # The third topic flynn, immunity, nana


    # [(0, '-0.190*"trump" + -0.136*"clinton" + -0.095*"hillary" + -0.094*"obama" + -0.089*"president" + -0.087*"woman" + 
    # -0.078*"republican" + -0.077*"party" + -0.077*"flynn" + -0.074*"candidate"'), (1, '-0.325*"boiler" + -0.284*"acr" + 
    # -0.244*"room" + -0.240*"pm" + -0.186*"broadcast" + -0.180*"radio" + -0.142*"tune" + -0.142*"animal" + -0.134*"jay" + 
    # -0.132*"episode"'), (2, '-0.623*"flynn" + -0.182*"immunity" + -0.122*"nana" + -0.116*"mr" + -0.110*"30" + -0.108*"march" 
    # + -0.102*"russian" + -0.100*"source" + 0.095*"school" + -0.092*"adviser"'), (3, '0.217*"clinton" + -0.186*"school" 
    # + -0.176*"student" + 0.141*"hillary" + -0.121*"county" + -0.121*"flynn" + 0.110*"sander" + 0.098*"debate" + 0.097*"woman" 
    # + 0.091*"nominee"'), (4, '0.200*"email" + -0.194*"trump" + 0.168*"dnc" + -0.142*"flynn" + 0.126*"clinton" + -0.112*"cruz" 
    # + 0.111*"department" + 0.110*"rich" + 0.102*"wikileaks" + 0.099*"sander"'), (5, '0.276*"student" + 0.160*"conference" 
    # + 0.160*"school" + 0.137*"trump" + -0.125*"mccain" + -0.124*"obama" + 0.105*"flynn" + 0.102*"campus" + 0.102*"yearbook" 
    # + -0.101*"putin"'), (6, '-0.349*"conference" + 0.193*"flynn" + -0.187*"press" + -0.171*"mark" + -0.166*"sean" + -0.166*"levin" 
    # + -0.166*"hannity" + -0.142*"discussing" + -0.135*"iowa" + -0.112*"immigration"')]



# =============================================================================
# Text Classification
# =============================================================================

# Can we create a custom classifier to accurately classify fake news versus factual news in our dataset?

# print(data.head())

# We have our text clean column that's been appropriately tokenized lemmatized converted into lowercase stopwords have been removed.

#                                                title                                               text  ... vader_sentiment_score vader_sentiment_label
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...  ...               -0.3660              negative
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...  ...               -0.8197              negative
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...  ...                0.9779              positive
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  ...               -0.3400              negative
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...  ...                0.8590              positive

# [5 rows x 7 columns]



# Creating our x and y variables to split into training and test data

# This is going to be our clean text that we're then going to go forward and vectorize and give to our algorithm as its input

x = [' '.join(map(str, tokens)) for tokens in data['text_clean']]

# Our Y value is just going to be our fake or factual tag

y = data['fake_or_factual']

# Text vectorization


# Create a CountVectorizer, which uses the Bag of Words model to convert text into numerical features

countvec = CountVectorizer()

# Learn the vocabulary from x and transform the text into a document-term matrix

countvec_fit = countvec.fit_transform(x)

# Convert the document-term matrix into a pandas DataFrame

# Converting our countvec_fit to an array

# This will successfully vectorized our text and it's ready to be split into training and test data

bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns=countvec.get_feature_names_out())


# Creating our xTrain, xTest, yTrain, yTest to split up our training data

# Split our bag of words dataset and our y vector

# test_size = 0.3 meaning 30% of the data is going to be going to our test dataset and won't be used for training

xTrain, xTest, yTrain, yTest = train_test_split(
    bag_of_words,
    y,
    test_size=0.3,
    random_state=0,
    stratify=y,
)

# Creating our classifier with just a simple logistic regression, to see how it performs, and use it as a baseline for other models we may want to try.

# Reference our xTrain and yTrain and our logistic regression model has now been created
# We want to use this to predict our test data set and compare this to our actuals.
lr = LogisticRegression(random_state=0).fit(xTrain, yTrain)

y_pred_lr = lr.predict(xTest)

# Computing the accuracy score, we will be using the accuracy score function and comparing the accuracy between y_pred_lr, the predicted labels from our logistic regression model,
# and comparing this to our actuals in yTest.

# We have some really good accuracy at 93%

# accuracy_score(yTest, y_pred_lr)
# print(accuracy_score(yTest, y_pred_lr)) # 0.9333333333333333

# print(classification_report(yTest, y_pred_lr))

# This is the chart for 0.8666666666666667 

#               precision    recall  f1-score   support

# Factual News       0.83      0.89      0.86        28
#    Fake News       0.90      0.84      0.87        32

#     accuracy                           0.87        60
#    macro avg       0.87      0.87      0.87        60
# weighted avg       0.87      0.87      0.87        60


# Using SVM support vector machine

# Creating our model using the SGD classifier, and then we reference .fit and our xTrain and yTrain

svm = SGDClassifier(random_state=0).fit(xTrain, yTrain)

# Running our predictions using this new model

y_pred_svm = svm.predict(xTest)

# print(accuracy_score(yTest, y_pred_svm)) # 0.9

# As we can see the model has not performed as well as the logistic regression.

# This is why it's really important to try simple models first before moving to anything more complex

# Printing the classification report

# print(classification_report(yTest, y_pred_svm))


#               precision    recall  f1-score   support

# Factual News       0.88      0.91      0.89        32
#    Fake News       0.89      0.86      0.87        28

#     accuracy                           0.88        60
#    macro avg       0.88      0.88      0.88        60
# weighted avg       0.88      0.88      0.88        60

# The model has worked well and it's giving us 90% accuracy, however the logistic regression has seemed to work better.

# So to answer the question, can we create a classifier that can classify different news as fake or factual? 

# The answer is yes, and it seems to be quite a good problem for these algorithms. They seem to work with really good accuracy.