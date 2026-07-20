import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # Scikit-learn is a powerful machine learning library in Python, and
                                                             # CountVectorizer is one of its tools for text processing. It transforms
                                                             # a collection of text documents into a matrix of token counts. In other words,
                                                             # it breaks the text into words (tokens) and counts how often each word appears.
                                                             # This gives us the numerical representation we need to apply machine learning
                                                             # techniques to text data.


data = [
    ' Most shark attacks occur about 10 feet from the beach since that is where the people are',
    'the efficiency with which he paired the socks in the drawer was quite admirable',
    'carol drank the blood as if she were a vampire',
    'giving directions that the mountains are to the west only works when you can see them',
    'the sign said there was road work ahead so he decided to speed up',
    'the gruff old man sat in the back of the bait shop grumbling to himself as he scooped out a handful of worms'
]


# Creating a Bag of Words representation is straightforward.

# The first step is to initialize the CountVectorizer class.

# By setting binary=True, every non-zero count becomes a one, which gives us
# the simple "word exists or does not exist" version of the Bag of Words model.

countvec = CountVectorizer(binary=True)  # This sets up CountVectorizer with the binary parameter enabled,
                                         # ready to learn the vocabulary from our text data.


# Fit means that CountVectorizer looks through the text and learns which unique
# words appear in the data.
# Transform then converts the text into numbers by creating a matrix that
# counts how often each word occurs.
# By using fit_transform(), we combine these two steps into one line of code.
# Now the variable countvec_fit holds our Bag of Words representation,
# a numerical version of the text.

countvec_fit = countvec.fit_transform(data)  # Apply CountVectorizer to our text data.
                                             # To do this, we call the fit_transform()
                                             # method on our CountVectorizer object.

# Visualize this representation as a table by converting it into a pandas
# DataFrame.

bag_of_words = pd.DataFrame(
    countvec_fit.toarray(),
    columns=countvec.get_feature_names_out()
)  # We apply the toarray() method to our countvec_fit variable to obtain a
   # regular two-dimensional array of numbers, which can then be placed into a
   # DataFrame. This way, every row becomes a document and every column becomes
   # a word from the vocabulary. The entries show the word counts.
   #
   # To label the columns, we use the get_feature_names_out() method from our
   # CountVectorizer. This method returns the actual words in the vocabulary
   # that CountVectorizer learned during the fit step.
   #
   # These words become our column headers, so each number in the table can be
   # clearly linked to a specific word.
   #
   # We can then print out this Bag of Words representation and see what it
   # looks like.

print(bag_of_words)


# Each row represents a different piece of text within our dataset.

# For example, row 0 is the text:
# "Most shark attacks occur about 10 feet from the beach since that is where
# the people are."
#
# We can see that the words "10", "about", and "are" are all present within
# this text.
#
# The next row down is simply the next piece of text in our dataset. Each row
# relates to one piece of text, and each column relates to one individual word.
# The values show the frequency of occurrence of each word in the document.
#
# If we want the table to indicate only whether a word is present or absent,
# without counting how many times it appears, we can do that too.
# CountVectorizer has a parameter called binary.

#    10  about  admirable  ahead  are  as  ...  which  with  work  works  worms  you
#0   1      1          0      0    1   0  ...      0     0     0      0      0    0
#1   0      0          1      0    0   0  ...      1     1     0      0      0    0
#2   0      0          0      0    0   1  ...      0     0     0      0      0    0
#3   0      0          0      0    1   0  ...      0     0     0      1      0    1
#4   0      0          0      1    0   0  ...      0     0     1      0      0    0
#5   0      0          0      0    0   1  ...      0     0     0      0      1    0

# [6 rows x 71 columns]


# The Bag of Words method is quick to build and straightforward to understand,
# but it has an important limitation: it only counts words. It doesn't
# consider their order or how important one word might be compared to another.
# While it is useful as a first step, it loses much of the context of the text.