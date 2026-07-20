from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =============================================================================
# Rule-Based Sentiment Analysis
# =============================================================================
#
# Rule-based sentiment analysis uses predefined rules that connect words to
# emotions or attitudes.
#
# For example:
#
#   "great"  -> Positive (+0.8)
#   "sad"    -> Negative (-0.7)
#
# Each word is assigned a polarity score ranging from -1 to 1.
#
#   Positive  -> > 0
#   Neutral   -> 0
#   Negative  -> < 0
#
# The individual word scores are combined to calculate the overall sentiment
# of a sentence or document.
#
# Advantages
# ----------
# • Easy to understand
# • Fast to implement
# • Great starting point for sentiment analysis
#
# Limitations
# -----------
# • Cannot reliably detect sarcasm or irony
# • Different sentiment libraries may produce different results because they
#   use different lexicons and scoring methods.
#
# In this lesson we compare:
#
# • TextBlob
# • VADER
# =============================================================================



# =============================================================================
# Example Sentences
# =============================================================================

sentence_1 = "I had a great time at the movie it was really fun"
sentence_2 = "I had a great time at the movie but the parking was terrible"
sentence_3 = "I had a great time at the movie but the parking wasn't great"
sentence_4 = "I went to see a movie"



# =============================================================================
# TextBlob Sentiment Analysis
# =============================================================================

print("----- TextBlob Sentiment Analysis -----\n")

print(sentence_1)

sentiment_score_1 = TextBlob(sentence_1)
print(sentiment_score_1.sentiment.polarity)
# 0.55 -> Moderately positive sentiment


print(sentence_2)

sentiment_score_2 = TextBlob(sentence_2)
print(sentiment_score_2.sentiment.polarity)
# -0.10 -> Close to neutral because the sentence contains both
# positive and negative sentiment.


print(sentence_3)

sentiment_score_3 = TextBlob(sentence_3)
print(sentiment_score_3.sentiment.polarity)
# 0.80


print(sentence_4)

sentiment_score_4 = TextBlob(sentence_4)
print(sentiment_score_4.sentiment.polarity)
# 0.0 -> Neutral sentiment



# =============================================================================
# VADER Sentiment Analysis
# =============================================================================

print("\n----- VADER Sentiment Analysis -----\n")

vader_sentiment = SentimentIntensityAnalyzer()


print(sentence_1)

print(vader_sentiment.polarity_scores(sentence_1))

# Returns a dictionary containing:
#
# neg      -> Negative score
# neu      -> Neutral score
# pos      -> Positive score
# compound -> Overall sentiment score (-1 to 1)


print(sentence_2)

print(vader_sentiment.polarity_scores(sentence_2))
# More negative than TextBlob.


print(sentence_3)

print(vader_sentiment.polarity_scores(sentence_3))

# VADER assigns a negative compound score whereas TextBlob gives
# the sentence a positive score.
#
# This suggests VADER handles negation
# (e.g. "wasn't great") better for this example.


print(sentence_4)

print(vader_sentiment.polarity_scores(sentence_4))
# Neutral sentiment.



# =============================================================================
# Key Takeaway
# =============================================================================
#
# It is a good idea to compare multiple sentiment analysis libraries on a sample
# of your data.
#
# Different tools interpret language differently, and testing multiple packages
# helps determine which one performs best for your specific dataset and use case.
# =============================================================================