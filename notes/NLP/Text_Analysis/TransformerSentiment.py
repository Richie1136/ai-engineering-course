from transformers import pipeline


# =============================================================================
# Transformer-Based Sentiment Analysis
# =============================================================================
#
# Modern sentiment analysis often relies on transformer models.
#
# Transformers are based on deep learning and are designed to understand how
# words influence and relate to each other, even when they appear far apart
# in a sentence.
#
# Unlike rule-based methods, transformers understand context. This allows them
# to handle more complex sentences, subtle tones, and sometimes sarcasm more
# effectively.
# =============================================================================


# =============================================================================
# Example Sentences
# =============================================================================

sentence_1 = "I had a great time at the movie it was really fun"
sentence_2 = "I had a great time at the movie but the parking was terrible"
sentence_3 = "I had a great time at the movie but the parking wasn't great"
sentence_4 = "I went to see a movie"


# =============================================================================
# Default Transformer Sentiment Model
# =============================================================================

print("----- Default Transformer Sentiment Model -----\n")

sentiment_pipeline = pipeline("sentiment-analysis")

print(sentence_1)
print(sentiment_pipeline(sentence_1))

print(sentence_2)
print(sentiment_pipeline(sentence_2))

print(sentence_3)
print(sentiment_pipeline(sentence_3))

print(sentence_4)
print(sentiment_pipeline(sentence_4))

# The default model correctly identifies the first three examples.
# However, it classifies sentence_4 as positive, even though it is more neutral.
#
# This shows that even powerful transformer models can misclassify text.


# =============================================================================
# Specific Pre-Trained Sentiment Model
# =============================================================================

print("\n----- Specific Pre-Trained Sentiment Model -----\n")

specific_model = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis"
)

print(sentence_1)
print(specific_model(sentence_1))

print(sentence_2)
print(specific_model(sentence_2))

print(sentence_3)
print(specific_model(sentence_3))

print(sentence_4)
print(specific_model(sentence_4))

# By switching to a model that is better suited to the use case, we can sometimes
# achieve more accurate results.
#
# In this example, the specific model classifies sentence_4 as neutral.


# =============================================================================
# Key Takeaway
# =============================================================================
#
# Transformer models are powerful because they understand context better than
# rule-based sentiment tools.
#
# However, model choice still matters. Different pre-trained models can produce
# different results, so it is important to test models on examples that are
# similar to your actual dataset.
# =============================================================================