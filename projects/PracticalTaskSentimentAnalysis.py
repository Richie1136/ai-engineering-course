import pandas as pd  # A library for storing and working with structured data.
import re  # For regular expressions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt

# VADER is a rule-based method of sentiment analysis, which means it uses a
# predefined set of rules and a built-in dictionary of words to determine
# whether the text expresses a positive, negative, or neutral emotion.

# Using read_csv to load the book reviews

data = pd.read_csv('../notes/data/book_reviews_sample.csv')

# print(data.head())  # Grab the headers in the data

# print(data['reviewText'][0])  # Grab first entry in the reviewText column

# For sentiment analysis, small words like "not" or "very" can completely
# change the meaning of a sentence. If we removed or altered them, our
# sentiment results could become inaccurate. We can usually leave punctuation
# in, but for our example, we'll simply remove it to keep things clean
# and consistent.

# We take the original reviewText column and apply the str.lower() method,
# which converts all the text to lowercase.

data['reviewText_clean'] = data['reviewText'].str.lower()

# Next, we remove punctuation. We use the apply() method to go through each row
# in our dataset and apply a small lambda function. Inside the lambda, we call
# re.sub(), which searches for punctuation marks and replaces them with
# an empty string.
#
# Finally, we print the first few rows again to verify that the text has been
# converted to lowercase and the punctuation has been removed.

data['reviewText_clean'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['reviewText_clean']), axis=1)

# print(data['reviewText_clean'][0])  # Printing the first item in the reviewText_clean column, lowercase and no punctuation

# print(data.head())  # Our new reviewText_clean column contains text that is
# all lowercase and free of punctuation.

# Create a VADER sentiment analyzer.
# We don't need to pass in any arguments because VADER already comes with a
# predefined sentiment lexicon, a built-in list of words and their sentiment
# scores. In other words, everything it needs to analyze text is already
# included.

vader_sentiment = SentimentIntensityAnalyzer()

# This object contains all the built-in rules and word scores that VADER uses
# to calculate sentiment. Note that if we were using TextBlob, this step would
# look different.

# TextBlob doesn't require a separate analyzer object. Instead, we create a
# TextBlob object for each piece of text that we want to analyze by placing
# the text inside TextBlob parentheses.

# For example:

# textblob_sentiment = TextBlob(text_1)

# Creating a new column in our dataframe to store the sentiment scores.

# For each row in our dataset, we'll use our VADER Sentiment Analyzer and call
# its polarity_scores() method. This method returns a dictionary containing
# four sentiment scores: negative, neutral, positive, and compound.
#
# We only need the compound score, which summarizes the overall sentiment of
# the review on a scale from -1 to 1. That value will be stored in our new
# column, giving us one sentiment score per review.

data['vader_sentiment_scores'] = data['reviewText_clean'].apply(lambda review: vader_sentiment.polarity_scores(review)['compound'])

print(data.head())  # Each review now has a corresponding VADER sentiment score.

# Bins - The value ranges that define how the scores are divided.

# Scores from -1 to -0.1 are classified as negative.
# Scores from -0.1 to 0.1 are classified as neutral.
# Scores from 0.1 to 1 are classified as positive.

bins = [-1, -0.1, 0.1, 1]

# Labels corresponding to each sentiment range

names = ['negative', 'neutral', 'positive']

# pd.cut - The first argument is the column we want to divide into ranges.
# The bins parameter defines the numerical boundaries of those ranges.
# Each bin represents an interval of scores.
# The labels parameter specifies the names we want to assign to each bin.
#
# In our example, pd.cut() looks at every sentiment score, figures out which
# interval or bin it falls into, and then assigns it the label negative,
# neutral, or positive based on that range.

data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_scores'], bins, labels=names)

print(data.head())  # The new column classifies each review as negative, neutral, or positive.

# Making it a bar chart

# Call value_counts() to count how many reviews fall into each sentiment
# category: positive, neutral, and negative.

# The plot part calls pandas' built-in plotting features, which work together
# with matplotlib, a popular plotting library in Python.

# plot.bar() tells pandas to draw a bar chart instead of another type of
# visualization, such as a line or pie chart. Each bar represents one
# sentiment category, and its height shows how many reviews belong to
# that category.

# data['vader_sentiment_label'].value_counts().plot.bar()

# Added the below 5 lines to style the chart and show the bar graph

# plt.xlabel("Sentiment")
# plt.ylabel("Number of Reviews")
# plt.title("VADER Sentiment Distribution")
# plt.tight_layout()
# plt.show()

# Basic sentiment analysis pipeline
transformer_pipeline = pipeline('sentiment-analysis')

transformer_labels = []

# First, we pass each review into our transformer pipeline. The pretrained
# model analyzes the text and returns a list containing the prediction.
#
# Next, we extract the sentiment label from the first item in that list.
#
# Finally, we append the label to our transformer_labels list so that we can
# store the prediction for every review in the dataset.

for review in data['reviewText_clean'].values:
    sentiment_result = transformer_pipeline(review)
    sentiment_label = sentiment_result[0]['label']
    transformer_labels.append(sentiment_label)

# Created a new column to store all transformer labels

data['transformer_sentiment_label'] = transformer_labels

# data['transformer_sentiment_label'].value_counts().plot.bar()

# plt.show()

# Unlike VADER, the default Hugging Face sentiment-analysis pipeline only
# predicts two classes: POSITIVE and NEGATIVE. It does not include a neutral
# category, so our bar chart only contains two sentiment labels.