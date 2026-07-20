import pandas as pd # Pandas will help us organize the tagged text into a clear format so we can analyze it more easily.
import matplotlib.pyplot as plt # A module inside Matplotlib that provides functions for creating and displaying plots.
import seaborn as sns # A Python data visualization library built on top of Matplotlib
import spacy # comes with powerful pre-trained models.
# displacy - built-in visualization tool that lets you see entities highlighted directly in text
from spacy import displacy, tokenizer # tokenzier - breaks text into tokens so spaCy can process it correctly
import re # imports Python's built-in re (Regular Expressions) module
import nltk 
from nltk.tokenize import word_tokenize # imports the word tokenizer, which splits text into individual words and punctuation.
# PorterStemmer - A stemmer reduces words to their root form by chopping off endings.
from nltk.stem import PorterStemmer, WordNetLemmatizer # WordNetLemmatizer - A lemmatizer also reduces words to their base form, but it uses a dictionary and linguistic rules.
from nltk.corpus import stopwords # This imports a list of common words that often don't add much meaning.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # rule-based sentiment analysis tool that determines whether text is positive, negative or neutral
import gensim # NLP library focused on working with large collections of text.
import gensim.corpora as corpora # A corpus is simply a collection of documents. This module converts text into a format Gensim models understand.
from gensim.models.coherencemodel import CoherenceModel # Used to evaluate topic models. It measures how meaningful the generated topics are.
# LsiModel - It discovers hidden relationships between words and documents.
from gensim.models import LsiModel, TfidfModel # TfidfModel - highlights words that are important within a document while reducing the weight of very common words.
# TfidfVectorizer - It converts text into numerical features that machine learning models can use.
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # CountVectorizer - Creates a Bag of Words representation. It simply counts how many times each word appears.
from sklearn.model_selection import train_test_split # Splits your dataset into Training data and Testing data
# LogisticRegression - A supervised machine learning algorithm used primarily for classification. It predicts categories rather than continuous values.
from sklearn.linear_model import LogisticRegression, SGDClassifier # SGDClassifier - It trains linear classifiers efficiently, especially on large datasets.
# accuracy_score - Measures the percentage of correct predictions.
from sklearn.metrics import accuracy_score, classification_report # classification_report - Produces a detailed evaluation of your classifier.


# Going to be running through a practical example, taking one data set and running through all the steps that we've covered in tbis course in a real life
# business setting.


# Imagine you're working for a social media company, and the company is concerned with the growing amount of fake news circulating on its platofrm.


# They've assigned you, as a data scientist, to investigate how fake news can be recognized, and create a method of identifying it.


# First step is to explore and clean the data, and then working to classify fake versus factual news stories.

# We'll also create some plots of our outputs and discuss how we will communicate our findings to stakeholders.

# Using pandas package for data manipulation, matplotlib and seaborn for plotting, spacy, re, NLTK, gensim and sklearn for all different types of analysis that we want to do.

# Going to set some plot options to get started.

# Going to set the figure size as 12.8 to make sure all of our charts are printed in a nice size, and we'll also specify a default plot color to use.


# Set plot options
plt.rcParams['figure.figsize'] = (12,8)
default_plot_color = '#00bfbf'


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


# POS Tagging

# Because one of our tasks is to determine the differences between fake and factual news, we want to split out the dataset into fake news and factual news.

# And then we can compare the POS tags that occur between each of the different data dets.

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
    return [(i.text, i.ent_type_, i.pos_) for i in doc] # We're not only extracting the POS tags, but the int type for named entity recognition


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
# that any kind of really frequently occuring words that you don't want in your data set are properly included within the stop words.

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



# Named Entities

# It is better to be look at the named entities before you do any kind of text pre-processing.

# We are doing it now, to give our models the best chance of pulling out those interesting named entities before we go in and do any kind of 
# pre processing and cleaning of the data. Because we already pulled out the named entity tags in our last lesson when we did POS tagging, we don't have to do that step again,
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

# So for example, if we've got different people being pulled out of the different data sets, we want to make sure that it's going to be represented by the sane
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


# Text Pre-Processing

# So far, we've loaded in our data set and explored it a little using parts of speech tagging and named entity recognition.


# print(data.head()) # title column, text column, the date, and the fake or factual tags

#                                                title                                               text                 date fake_or_factual
#0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News
#1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News
#2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News
#3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News
#4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News


# Creating regular expression to look for the hypen in the text column and remove everything before the first hypen


# Creating new column that is going to contain all of our cleaned up text.

# Using data.apply to apply our re.sub function over the rows in our data set
# In our regular expression we are looking for the first hypen in the text and remove the hypen and everything before it.
# We specify the regular expression syntax and we want to replace this with blank, so essentially remove it from our text.
# Then we specify we want to run this over the text column and our axis = 1
data['text_clean'] = data.apply(lambda x: re.sub(r"^[^-]*-\s", "", x['text']), axis=1)

print(data.head())

# Our text clean column has been added and the location tags have been removed.

#                                                title                                               text                 date fake_or_factual                                         text_clean
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News  There are two small problems with your analogy...
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News  Buried in Trump s bonkers interview with New Y...
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News  Women make up over 50 percent of this country,...
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News  U.S. Defense Secretary Jim Mattis said on Mond...
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News  Northern Ireland s political parties are rapid...



# Converting our text into lowercase

data['text_clean'] = data['text_clean'].str.lower()

# Remove punctuation

# Inside the regex looking for anything that isn't a word or a space and removing it

data['text_clean'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['text_clean']), axis=1)

print(data.head())

#                                                title                                               text                 date fake_or_factual                                         text_clean
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News  there are two small problems with your analogy...
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News  buried in trump s bonkers interview with new y...
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News  women make up over 50 percent of this country ...
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News  us defense secretary jim mattis said on monday...
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News  northern ireland s political parties are rapid...


# Removing stop words

# When we did POS tagging above we had really common tokens that were mostly stopwords, so its really important to go back and sync that up with your
# stop words list and make sure that all those really common occuring words that you can remove are in that stop words list to be removed.

en_stopwords = stopwords.words('english')

data['text_clean'] = data['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

print(data.head())

#                                                title                                               text                 date fake_or_factual                                         text_clean
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News  two small problems analogy susan jesus muslim ...
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News  buried trump bonkers interview new york times ...
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News  women make 50 percent country grossly underrep...
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News  us defense secretary jim mattis said monday ru...
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News  northern ireland political parties rapidly run...



# Tokenizing the text

# This will go through each of our rows in our data set
# Take the text from text_clean and converts it into our word tokens.

data['text_clean'] = data.apply(lambda x: word_tokenize(x['text_clean']), axis=1)


# Lemmatizing

# Going to be using Lemmatizing instead of stemming, because i want to use this more intelligent method to keep alot of the context
# and meaning of the words where possible
# 

lemmatizer = WordNetLemmatizer()

# We use lemmatizer.lemmatize and in the brackets specify our tokens. for token in tokens

data['text_clean'] = data['text_clean'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

print(data.head())


#                   title                                               text                 date fake_or_factual                                         text_clean
# 0  HOLLYWEIRD LIB SUSAN SARANDON Compares Muslim ...  There are two small problems with your analogy...         Dec 30, 2015       Fake News  [two, small, problem, analogy, susan, jesus, m...
# 1   Elijah Cummings Called Trump Out To His Face ...  Buried in Trump s bonkers interview with New Y...        April 6, 2017       Fake News  [buried, trump, bonkers, interview, new, york,...
# 2   Hillary Clinton Says Half Her Cabinet Will Be...  Women make up over 50 percent of this country,...       April 26, 2016       Fake News  [woman, make, 50, percent, country, grossly, u...
# 3  Russian bombing of U.S.-backed forces being di...  WASHINGTON (Reuters) - U.S. Defense Secretary ...  September 18, 2017     Factual News  [u, defense, secretary, jim, mattis, said, mon...
# 4  Britain says window to restore Northern Irelan...  BELFAST (Reuters) - Northern Ireland s politic...   September 4, 2017     Factual News  [northern, ireland, political, party, rapidly,...


# Creating a list of our clean tokens and then go on to look at the Unigrams