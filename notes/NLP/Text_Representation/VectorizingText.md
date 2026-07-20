# Vectorizing Text

This is the third step for getting the data in the right format for machine learning.

This is where text vectorization comes in.

Passing clean text into an algorithm isn't enough. We need to convert this into a numerical representation that the machine learning algorithm can understand.

## Two of the Most Common Methods

### 1. Bag of Words Model

It just counts which of our words appear in which of our documents. It's simple and easy to understand, but you do lose a lot of context.

### 2. Term Frequency-Inverse Document Frequency (TF-IDF)

It calculates the importance of that word for that particular document and takes into account how that word also appears in each of the other documents in our data.
