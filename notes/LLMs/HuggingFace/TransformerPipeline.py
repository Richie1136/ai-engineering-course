from transformers import pipeline


# The Transformers library provides a simple way to start working with
# pre-trained language models through the pipeline() function.

# A pipeline handles much of the work for us, such as preprocessing the text,
# running the model, and returning an easy-to-understand output.

# With just a few lines of code, we can connect to a language model, provide
# an input, and receive a prediction.


# =====================================================
# Sentiment Analysis Pipeline
# =====================================================

# Create a pipeline for sentiment analysis.

# The string 'sentiment-analysis' tells the pipeline which task we want to
# perform. A default pre-trained model is automatically selected.

sentiment_classifier = pipeline("sentiment-analysis")

# Run the sentiment classifier on a new piece of text.

sentiment_classifier(
    "I'm so excited to be learning about large language models."
)

# This single line of code returns the predicted sentiment together with the
# model's confidence score.

# print(sentiment_classifier("I'm so excited to be learning about large language models."))

# [{'label': 'POSITIVE', 'score': 0.9995085000991821}]


# =====================================================
# Named Entity Recognition (NER)
# =====================================================

# Each pipeline has a default model, but we can also specify which model we
# want to use.

# Here, we create a Named Entity Recognition (NER) pipeline using the
# dslim/bert-base-NER model.

ner = pipeline(
    "ner",
    model="dslim/bert-base-NER"
)

# print(ner("Her name is Anna, and she works in New York City for Morgan Stanley."))

# [{'entity': 'B-PER', 'score': 0.9943198, 'index': 4, 'word': 'Anna', 'start': 12, 'end': 16},
#  {'entity': 'B-LOC', 'score': 0.999587, 'index': 10, 'word': 'New', 'start': 35, 'end': 38},
#  {'entity': 'I-LOC', 'score': 0.9994042, 'index': 11, 'word': 'York', 'start': 39, 'end': 43},
#  {'entity': 'I-LOC', 'score': 0.99958605, 'index': 12, 'word': 'City', 'start': 44, 'end': 48},
#  {'entity': 'B-ORG', 'score': 0.9972107, 'index': 14, 'word': 'Morgan', 'start': 53, 'end': 59},
#  {'entity': 'I-ORG', 'score': 0.9980744, 'index': 15, 'word': 'Stanley', 'start': 60, 'end': 67}]

# The pipeline abstracts away much of the complexity involved in working with
# language models, such as preprocessing the text input and formatting the
# model's output into an interpretable result.

# The Hugging Face Model Hub contains information about the many models that
# are available. You can search for specific models or browse models by task.


# =====================================================
# Zero-Shot Classification
# =====================================================

# Zero-shot learning is when a model performs a task without receiving any
# additional training for that specific task.

# The general knowledge learned during pre-training is enough for the model to
# make predictions.

# In this example, we'll classify a sentence without explicitly training the
# model on our chosen labels.

zeroshot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

sequence_to_classify = "One day I will see the world."

candidate_labels = [
    "travel",
    "cooking",
    "dancing"
]

# Run the zero-shot classifier using our sequence and candidate labels.

# print(zeroshot_classifier(sequence_to_classify, candidate_labels))

# {'sequence': 'One day I will see the world.',
#  'labels': ['travel', 'dancing', 'cooking'],
#  'scores': [0.9938650727272034, 0.003273779060691595, 0.00286103505641222]}

# The output contains three main pieces of information:
#
# - The original sequence.
# - The predicted labels.
# - A confidence score for each label.

# The model predicts that the sentence "One day I will see the world." is most
# closely related to the "travel" label.

# Notice that we did not provide any additional training data. The model uses
# the general knowledge learned during pre-training to classify the text.

# This example demonstrates how easy it is to begin working with language
# models using the Transformers pipeline.

# The Transformers library provides many different pipelines for a wide
# variety of natural language processing tasks.