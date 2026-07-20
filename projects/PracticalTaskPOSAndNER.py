# PracticalTaskPOSAndNER.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
import pandas as pd
import matplotlib.pyplot as plt


# Dataset of BBC news articles.
# Load the BBC data into Python using pandas read_csv.

bbc_data = pd.read_csv('../notes/data/bbc_news.csv')
bbc_data.head()

# print(bbc_data.head())

# bbc_data.info()  # Get more details about our dataset.

# Extract the title column from the bbc_data DataFrame.

titles = pd.DataFrame(bbc_data['title'])
titles.head()

# print(titles.head())  # One column containing the titles of the news articles.

# Convert the text to lowercase.

titles['lowercase'] = titles['title'].str.lower()

# print(titles.head())

# Stop word removal.

en_stopwords = stopwords.words('english')

# Loop through every word and keep only the ones that aren't in the stopword list.

titles['no_stopwords'] = titles['lowercase'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in en_stopwords])
)

# print(titles)

# Punctuation removal.

# Remove symbols such as commas, periods, or question marks.

titles['no_stopwords_no_punct'] = titles.apply(
    lambda x: re.sub(r'[^\w\s]', "", x['no_stopwords']),
    axis=1
)

# print(titles)

# Tokenization - Splits text into words.

# Take the original title column and apply the word_tokenize function to each row.
# For every row in our DataFrame, call it x, and then tokenize the title inside it.
# This creates a new column called token_raw, which contains lists of all the words
# in each original title, including punctuation.

titles['token_raw'] = titles.apply(lambda x: word_tokenize(x['title']), axis=1)

# Tokenize the cleaned version of each title, the one without stopwords or punctuation,
# and store that in a new column called token_clean.
# Now, instead of long strings of text, we have lists of individual words that Python can easily analyze.

titles['token_clean'] = titles.apply(lambda x: word_tokenize(x['no_stopwords_no_punct']), axis=1)

# Lemmatizing - Reduces a word to a meaningful base form while preserving its intended meaning.

lemmatizer = WordNetLemmatizer()

# Loop through each token, or each word in the list, and replace it with its base form.
# The result is a new column called tokens_clean_lemmatized, which stores the final,
# fully processed versions of our titles: lowercased, cleaned, tokenized, and lemmatized.

titles['tokens_clean_lemmatized'] = titles['token_clean'].apply(
    lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
)

# print(titles.head())

tokens_raw_list = sum(titles['token_raw'], [])
tokens_clean_list = sum(titles['tokens_clean_lemmatized'], [])

# POS Tagging

nlp = spacy.load('en_core_web_sm')

spacy_doc = nlp(' '.join(tokens_raw_list))  # spaCy document to store the text in the right format.

# The result stored in spacy_doc is a spaCy document object that contains the text along with
# detailed annotations, such as each word's part-of-speech tag.

pos_df = pd.DataFrame(columns=['token', 'pos_tag'])

# Create a new DataFrame to store the results of our part-of-speech tagging.

# There are two columns: one for the token, which is each individual word, and one for its
# POS tag, which shows the grammatical category assigned by spaCy. In other words, this
# DataFrame lets us easily see which word belongs to which part of speech.

# Create a for loop to go through each token in our spaCy document and extract its
# corresponding part-of-speech tag. For every token in the spaCy document, we take the
# actual word stored as token.text and its part-of-speech tag stored as token.pos_.
# The from_records function creates a one-row DataFrame from these two pieces of information,
# and pd.concat then adds that new row to the existing table. By the end of the loop, the
# table contains two columns: one with each word from the document and one with its
# corresponding part of speech.

for token in spacy_doc:
    pos_df = pd.concat([
        pos_df,
        pd.DataFrame.from_records([{'token': token.text, 'pos_tag': token.pos_}])
    ], ignore_index=True)


# Token frequency count

# We want to count how many times each word appears in our document and which part of speech
# it belongs to.

# We create a new DataFrame called pos_df_counts. We start by using the groupby method.
# We group the data by two columns: the token column, which contains the word itself, and
# the pos_tag column, which contains its part of speech. Next, we use the size method,
# which counts how many rows fall into each of those groups. In other words, it counts how
# often each token-tag pair occurs.

# Then we use reset_index because when we group the data, pandas automatically places the
# grouped columns into the index, and the output is no longer displayed like a regular
# DataFrame with normal columns.

# reset_index removes those group labels from the index and turns them back into regular columns.
# This gives us a clean table again, where token, pos_tag, and counts each have their own column.
# In the parentheses, name='counts' gives a clear name to the new column that holds the counts.
# Instead of just seeing unlabeled numbers, we now have a proper counts column.

# Finally, we sort the values so that the most frequent tokens appear at the top.

pos_df_counts = pos_df.groupby(['token', 'pos_tag']).size().reset_index(name='counts').sort_values(
    by='counts',
    ascending=False
)

# print(pos_df_counts.head(10))

nouns = pos_df_counts[pos_df_counts.pos_tag == 'NOUN'][:10]

# print(nouns)

verbs = pos_df_counts[pos_df_counts.pos_tag == 'VERB'][:10]

# print(verbs)

adj = pos_df_counts[pos_df_counts.pos_tag == 'ADJ'][:10]

# print(adj)


# Named Entity Recognition

# Create a new DataFrame, which will hold each token and its associated named entity tag.
# This includes tags like person, organization, or GPE for geopolitical entity.

# Initialize it with two columns: one for the token and one for the NER tag.

ner_df = pd.DataFrame(columns=['token', 'ner_tag'])

# Create a for loop that goes through every entity in our spaCy document using spacy_doc.ents.
# Inside the loop, we use a small condition. The function pd.isna checks if a value is missing.
# Here, it is checking if token.label_ exists. The phrase is False simply means we only continue
# if the value is not missing. So together, that line says if the token has a valid label, then
# process it.

# When that condition is true, we take the token text, which is the word itself, and token.label_,
# which gives us the human-readable version of the named entity tag.

# We then use from_records again to create a one-row DataFrame from this information and concat it
# to our main NER DataFrame. By the end, NER contains a clear list of all the named entities that
# spaCy found in the text.

for token in spacy_doc.ents:
    if pd.isna(token.label_) is False:
        ner_df = pd.concat([
            ner_df,
            pd.DataFrame.from_records([{'token': token.text, 'ner_tag': token.label_}])
        ], ignore_index=True)

# Most common named entities that appear in our dataset.

ner_df_counts = ner_df.groupby(['token', 'ner_tag']).size().reset_index(name='counts').sort_values(
    by='counts',
    ascending=False
)

print(ner_df_counts.head())