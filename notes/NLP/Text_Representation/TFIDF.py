import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # Scikit-learn provides the TfidfVectorizer class,
                                                             # which automatically performs TF-IDF calculations.


# Using TF-IDF to vectorize text can be a stronger approach than the Bag of
# Words model because it preserves more information about the importance of
# each word.
#
# TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a
# word is within a collection of text documents. It does this by looking at
# two things:
# 1. How often the word appears in a single document.
# 2. How common the word is across all documents.

data = [
    ' Most shark attacks occur about 10 feet from the beach since that is where the people are',
    'the efficiency with which he paired the socks in the drawer was quite admirable',
    'carol drank the blood as if she were a vampire',
    'giving directions that the mountains are to the west only works when you can see them',
    'the sign said there was road work ahead so he decided to speed up',
    'the gruff old man sat in the back of the bait shop grumbling to himself as he scooped out a handful of worms'
]

# Term Frequency (TF) refers to how many times a word appears in a single
# document. A document means one piece of text, such as a row in your dataset
# or an item in a list.
#
# This gives us a relative frequency, showing how important that word is within
# that specific document.

# Inverse Document Frequency (IDF) looks at the bigger picture. Instead of
# looking inside just one document, it looks across the entire collection of
# documents.
#
# Once calculated, common words receive a lower score because, although they
# appear frequently, they don't help distinguish one document from another.
#
# Less common words receive a higher score because they carry more weight in
# describing what a specific document is about. This helps retain more context
# about how words are used and which words are important to specific documents
# and to the dataset as a whole.

tfidfvec = TfidfVectorizer()

tfidfvec_fit = tfidfvec.fit_transform(data)  # Learn the vocabulary and transform our text into TF-IDF vectors.

# Convert the TF-IDF representation into a pandas DataFrame.
# We first convert the sparse matrix into a regular two-dimensional array using
# toarray(), then use get_feature_names_out() to create the column names.

tfidf_bag = pd.DataFrame(
    tfidfvec_fit.toarray(),
    columns=tfidfvec.get_feature_names_out()
)

print(tfidf_bag)

# Instead of just ones and zeros, we now see a range of numbers. Each value
# reflects how important a word is within a specific document.
#
# TF-IDF captures not only whether a word appears, but also how important that
# word is compared to the rest of the documents in the dataset.
#
# As a result, when we apply machine learning, we give our models more
# meaningful information to work with, helping them better understand patterns
# in text.

#          10     about  admirable     ahead  ...      work    works    worms      you
#0  0.257061  0.257061   0.000000  0.000000  ...  0.000000  0.00000  0.00000  0.00000
#1  0.000000  0.000000   0.293641  0.000000  ...  0.000000  0.00000  0.00000  0.00000
#2  0.000000  0.000000   0.000000  0.000000  ...  0.000000  0.00000  0.00000  0.00000
#3  0.000000  0.000000   0.000000  0.000000  ...  0.000000  0.27104  0.00000  0.27104
#4  0.000000  0.000000   0.000000  0.290766  ...  0.290766  0.00000  0.00000  0.00000
#5  0.000000  0.000000   0.000000  0.000000  ...  0.000000  0.00000  0.21782  0.00000

# [6 rows x 71 columns]