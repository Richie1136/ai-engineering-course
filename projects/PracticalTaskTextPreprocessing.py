import nltk
from nltk.tokenize import word_tokenize # # word_tokenize - Splits text into words.
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords # For words that don't add much meaning
import re # For regular expressions
import pandas as pd # A Python library for working with structured data. 
# The Porter stemmer is a classic rule-based algorithm that reduces English words to a simpler base.
# looks it up in the WordNet dictionary to return its proper form

data = pd.read_csv('../notes/data/tripadvisor_hotel_reviews.csv')
data.info()
data.head()
# print(data.head())
data['Review'][0]
# print(data['Review'][0])

# data['review_lowercase'] = data['Review']

# print(type(data['Review']))

data['review_lowercase'] = data['Review'].str.lower()
# print(data.head())

en_stopwords = stopwords.words('english')
en_stopwords.remove('not')

# Go through every review in this column and run the following set of instructions on it.
data['review_no_stop_words'] = data['review_lowercase'].apply(lambda x: ' '.join([word for word in x.split(" ") if word not in (en_stopwords)])) # The apply function lets us take one column and perform a custom operation on every value in it

# lambda function - Anonymous function, the function doesn't have a name

# lambda x: ' '.join([word for word in x.split(" ") if word not in (en_stopwords)]) # Take each review and split it into individual words using the split method, this turns the review into one long string of text into a list of separate words.

# Then for every word in that list, Python checks whether the word appears in our list of stop words, if the word is not a stop word then it is kept

# When your're pre-processing text, it is always worth making a new column for each of the steps in your preprocessing

# print(data['review_no_stop_words'][0])

# Axis equals 1 means go through the data row by row rather than column by column. Inside the apply function, we use another lambda function to describe exactly what should happen for each row.

# The operation we're performing is handled by the sub function, which is used to search for specific text patterns and replace them with something else.

# So in this case we are replacing * with "star"

# x['review_no_stop_words']) tells python that re.sub should be applied specifically to the text from 'review_no_stop_words' column. The resulting text, with punctuation handled is then stored in the new column 'review_no_stop_words_no_punc'

data['review_no_stop_words_no_punc'] = data.apply(lambda x: re.sub(r"[*]", "star", x['review_no_stop_words']), axis=1)

# print("Review No Stop Word Puc", data.head())
data['review_no_stop_words_no_punc'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['review_no_stop_words_no_punc']), axis=1)

# print(data.head())

# Splitting the review_no_stop_words_no_punc' text into a list of words.

data['tokenized'] = data.apply(lambda x: word_tokenize(x['review_no_stop_words_no_punc']), axis=1)
# print(data['tokenized'][0])

# Applying stemming to each token, reducing words to their root form and storing the stemmed tokens as a list.

ps = PorterStemmer()

data['stemmed'] = data['tokenized'].apply(lambda tokens: [ps.stem(token) for token in tokens])

# print(data.head())

# Converts each token in the list to its dictionary (base) form while preserving the word's intended meaning.

lemmatizer = WordNetLemmatizer()

data['lemmatized'] = data['tokenized'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
# print(data['lemmatized'][0])

# Right now, each row in the 'lemmatized' column contains a separate list of tokens
# Each review is stored as its own list of lemmatized words
# We need to combine these smaller lists into one long list that contains every token from all reviews

# In Python, when we use sum() with list instead of numbers, it joins them together

# The code starts with the empty list, which as a starting point. Then it goes through the lemmatized column and keeps adding or concatenating
# each review's list of tokens to that empty list.
# After it's done, we end up with one big list that contains all the lemmatized words from all reviews combined.

# sum() keeps adding each review's list of tokens to the empty list
tokens_clean = sum(data['lemmatized'], [])

unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts())
print(unigrams)

bigrams = (pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts())
print("Bigrams\n", bigrams)

trigrams = (pd.Series(nltk.ngrams(tokens_clean, 3)).value_counts())
print("Trigrams\n", trigrams)