import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# CountVectorizer converts text into a Bag of Words representation by counting
# how many times each word appears.

# TfidfVectorizer converts text into a TF-IDF representation by assigning each
# word a score based on its importance within a document and across the dataset.

# train_test_split divides the dataset into training and testing data.

# LogisticRegression is a machine learning algorithm that learns patterns in
# the data and predicts categories, also called classes.

# accuracy_score calculates the percentage of predictions that were correct.

# classification_report provides a more detailed evaluation of the model,
# including precision, recall, and F1-score.


# Logistic Regression is actually a classification algorithm. This means its
# goal is to predict categories, also called classes.

# An example of this type of prediction is whether a student will pass or fail
# an exam. This is different from regression algorithms, where the goal is to
# predict a continuous value, such as a student's exact exam score.

# Logistic Regression is a strong starting point when building a text
# classifier. It can be used on its own or as a baseline model to compare
# against more complex approaches later.

# In this example, we'll build a Logistic Regression model that learns from
# text data and classifies new sentences as having either positive or
# negative sentiment.


# =====================================================
# Create the Dataset
# =====================================================

# Here, we're creating a pandas DataFrame with two columns:
#
# - text: the sentence that the model will analyze
# - sentiment: the label assigned to the sentence, either positive or negative
#
# This dataset will serve as the training ground for our Logistic Regression
# model.

data = pd.DataFrame(
    [
        (
            "i love spending time with my friends and family",
            "positive"
        ),
        (
            "that was the best meal i've ever had in my life",
            "positive"
        ),
        (
            "i feel so grateful for everything i have in my life",
            "positive"
        ),
        (
            "i received a promotion at work and i couldn't be happier",
            "positive"
        ),
        (
            "watching a beautiful sunset always fills me with joy",
            "positive"
        ),
        (
            "my partner surprised me with a thoughtful gift and it made my day",
            "positive"
        ),
        (
            "i am so proud of my daughter for graduating with honors",
            "positive"
        ),
        (
            "listening to my favorite music always puts me in a good mood",
            "positive"
        ),
        (
            "i love the feeling of accomplishment after completing a challenging task",
            "positive"
        ),
        (
            "i am excited to go on vacation next week",
            "positive"
        ),
        (
            "i feel so overwhelmed with work and responsibilities",
            "negative"
        ),
        (
            "the traffic during my commute is always so frustrating",
            "negative"
        ),
        (
            "i received a parking ticket and it ruined my day",
            "negative"
        ),
        (
            "i got into an argument with my partner and we're not speaking",
            "negative"
        ),
        (
            "i have a headache and i feel terrible",
            "negative"
        ),
        (
            "i received a rejection letter for the job i really wanted",
            "negative"
        ),
        (
            "my car broke down and it's going to be expensive to fix",
            "negative"
        ),
        (
            "i'm feeling sad because i miss my friends who live far away",
            "negative"
        ),
        (
            "i'm frustrated because i can't seem to make progress on my project",
            "negative"
        ),
        (
            "i'm disappointed because my team lost the game",
            "negative"
        )
    ],
    columns=["text", "sentiment"]
)

# print(data)


# =====================================================
# Shuffle the Dataset
# =====================================================

# Shuffle the dataset so that the positive and negative sentences are mixed
# together instead of being grouped by sentiment.

# The sample() function is used with frac=1, which means we are selecting
# 100% of the rows but returning them in a randomized order.

# After shuffling, reset_index() numbers the rows neatly again.

# The drop=True argument tells pandas not to keep the old index as a separate
# column.

data = data.sample(frac=1, random_state=7).reset_index(drop=True)


# =====================================================
# Separate the Features and Labels
# =====================================================

# Prepare the inputs for our algorithm.

# x holds the text data, which contains the features the model will use.

# y holds the target labels from the sentiment column.

x = data["text"]
y = data["sentiment"]


# =====================================================
# Logistic Regression Using Bag of Words
# =====================================================

# The next step is text vectorization. This means converting our sentences
# into numbers so that the machine learning model can work with them.

# First, we'll use the Bag of Words approach with Scikit-learn's
# CountVectorizer class.

# Each unique word in our dataset becomes a column.

# Each sentence becomes a row.

# The values show how many times each word appears in each sentence.

countvec = CountVectorizer()

# fit_transform() performs two steps:
#
# 1. fit() learns the vocabulary from the text.
# 2. transform() converts each sentence into a numerical representation.

countvec_fit = countvec.fit_transform(x)

# Convert the Bag of Words matrix into a pandas DataFrame.

# toarray() converts the sparse matrix into a regular two-dimensional array.

# get_feature_names_out() returns the words from the vocabulary so that they
# can be used as the column names.

bag_of_words = pd.DataFrame(
    countvec_fit.toarray(),
    columns=countvec.get_feature_names_out()
)

# Printing the DataFrame shows the Bag of Words representation.

# print(bag_of_words)

# Each row corresponds to one sentence.

# Each column corresponds to one word in the vocabulary.

# The values show how many times each word appears in the sentence.

#     accomplishment  after  always  am  an  ...  we  week  who  with  work
# 0                0      0       0   0   0  ...   0     0    0     0     0
# 1                0      0       1   0   0  ...   0     0    0     1     0
# 2                0      0       0   0   0  ...   0     0    1     0     0
# 3                0      0       1   0   0  ...   0     0    0     0     0
# 4                0      0       0   0   0  ...   0     0    0     0     0
# 5                0      0       0   0   0  ...   0     0    0     0     0
# 6                0      0       0   0   0  ...   0     0    0     1     0
# 7                0      0       0   1   0  ...   0     1    0     0     0
# 8                0      0       0   0   1  ...   1     0    0     1     0
# 9                0      0       0   0   0  ...   0     0    0     0     1
# 10               0      0       0   0   0  ...   0     0    0     0     0
# 11               0      0       0   0   0  ...   0     0    0     0     0
# 12               0      0       0   0   0  ...   0     0    0     1     1
# 13               0      0       0   0   0  ...   0     0    0     0     0
# 14               0      0       0   0   0  ...   0     0    0     0     0
# 15               0      0       0   0   0  ...   0     0    0     1     0
# 16               0      0       0   1   0  ...   0     0    0     1     0
# 17               0      0       0   0   0  ...   0     0    0     0     0
# 18               1      1       0   0   0  ...   0     0    0     0     0
# 19               0      0       1   0   0  ...   0     0    0     0     0

# [20 rows x 118 columns]


# =====================================================
# Split the Bag of Words Data
# =====================================================

# Next, split the data into training and testing sets.

# The training set is the portion of the dataset the model uses to learn.

# The testing set is kept aside until after training so that we can evaluate
# how well the model performs on data it has not seen before.

# When we split the data, we separate it into features and labels.

# Features:
# The input data used by the model. In this case, the features are the
# sentences represented using Bag of Words.

# Labels:
# The answers that we want the model to learn. In this case, the labels are
# positive and negative sentiment values.

# The split creates four variables:
#
# x_train: training features
# x_test: testing features
# y_train: training labels
# y_test: testing labels

# test_size=0.3 means that 30% of the dataset is used for testing and 70% is
# used for training.

# A split like this gives the model enough data to learn from while still
# leaving a portion aside to evaluate its performance.

# Test sizes between 20% and 30% are often used in practice.

# random_state=7 makes the split reproducible, meaning we will get the same
# training and testing sets each time we run the code.

x_train, x_test, y_train, y_test = train_test_split(
    bag_of_words,
    y,
    test_size=0.3,
    random_state=7,
    stratify=y
)


# =====================================================
# Train the Bag of Words Model
# =====================================================

# Create and train the Logistic Regression model.

# The fit() method takes our training features and training labels and teaches
# the model to recognize patterns.

# The model looks at the Bag of Words features in x_train and learns how those
# features are connected to the sentiment labels in y_train.

bag_of_words_model = LogisticRegression(
    random_state=1
)

bag_of_words_model.fit(
    x_train,
    y_train
)


# =====================================================
# Evaluate the Bag of Words Model
# =====================================================

# Use the trained model to generate predictions for x_test.

# The model has not seen these sentences during training.

bag_of_words_predictions = bag_of_words_model.predict(x_test)

# Accuracy tells us the percentage of sentences that the model predicted
# correctly.

# The conventional order for accuracy_score() is:
#
# accuracy_score(true_labels, predicted_labels)

bag_of_words_accuracy = accuracy_score(
    y_test,
    bag_of_words_predictions
)

print("Bag of Words Accuracy:")
print(bag_of_words_accuracy)

# Our model may not perform well because the dataset is very small.

# However, it still provides a useful baseline model that can be compared
# against more complex approaches later.

# Example result:
# 0.16666666666666666

# Print a classification report.

# This provides more detail than accuracy alone by showing how the model
# performed for each sentiment class.

print("\nBag of Words Classification Report:")
print(
    classification_report(
        y_test,
        bag_of_words_predictions,
        zero_division=0
    )
)

# Precision:
# Out of all the sentences the model predicted as positive or negative, what
# proportion were actually correct?

# Recall:
# Out of all the sentences that were truly positive or negative, what
# proportion did the model correctly identify?

# F1-score:
# A value between 0 and 1 that combines precision and recall into one measure.

# A low F1-score means that the balance between precision and recall is poor.


# =====================================================
# Logistic Regression Using TF-IDF
# =====================================================

# Now we'll repeat the process using TF-IDF instead of Bag of Words.

# TF-IDF assigns each word a score based on how important it is within a
# particular sentence and how common it is across the entire dataset.

tfidfvec = TfidfVectorizer()

# fit_transform() learns the vocabulary and converts the sentences into
# TF-IDF numerical representations.

tfidfvec_fit = tfidfvec.fit_transform(x)

# Convert the TF-IDF matrix into a pandas DataFrame.

tfidf_words = pd.DataFrame(
    tfidfvec_fit.toarray(),
    columns=tfidfvec.get_feature_names_out()
)

# Printing the DataFrame shows the TF-IDF representation.

# print(tfidf_words)

# Each row represents one sentence.

# Each column represents one word.

# Each value represents the TF-IDF score of that word within that sentence.

#     accomplishment     after    always  ...       who      with      work
# 0         0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 1         0.000000  0.000000  0.000000  ...  0.000000  0.269044  0.000000
# 2         0.000000  0.000000  0.000000  ...  0.000000  0.201637  0.000000
# 3         0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 4         0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 5         0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 6         0.000000  0.000000  0.307572  ...  0.000000  0.242821  0.000000
# 7         0.000000  0.000000  0.000000  ...  0.000000  0.218636  0.000000
# 8         0.000000  0.000000  0.000000  ...  0.000000  0.291153  0.408704
# 9         0.358752  0.358752  0.000000  ...  0.000000  0.000000  0.000000
# 10        0.000000  0.000000  0.000000  ...  0.000000  0.232236  0.000000
# 11        0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.345355
# 12        0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 13        0.000000  0.000000  0.304510  ...  0.000000  0.000000  0.000000
# 14        0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 15        0.000000  0.000000  0.000000  ...  0.346057  0.000000  0.000000
# 16        0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 17        0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 18        0.000000  0.000000  0.000000  ...  0.000000  0.000000  0.000000
# 19        0.000000  0.000000  0.270609  ...  0.000000  0.000000  0.000000

# [20 rows x 118 columns]


# =====================================================
# Split the TF-IDF Data
# =====================================================

# Create four variables for the TF-IDF training and testing data:
#
# tfidf_x_train: training features
# tfidf_x_test: testing features
# tfidf_y_train: training labels
# tfidf_y_test: testing labels

# We use the same test_size and random_state as the Bag of Words model so that
# both approaches are evaluated using the same type of split.

tfidf_x_train, tfidf_x_test, tfidf_y_train, tfidf_y_test = train_test_split(
    tfidf_words,
    y,
    test_size=0.3,
    random_state=7,
    stratify=y
)


# =====================================================
# Train the TF-IDF Model
# =====================================================

# Create and train another Logistic Regression model using the TF-IDF
# features.

tfidf_model = LogisticRegression(
    random_state=1
)

tfidf_model.fit(
    tfidf_x_train,
    tfidf_y_train
)


# =====================================================
# Evaluate the TF-IDF Model
# =====================================================

# Generate predictions for the TF-IDF testing data.

tfidf_predictions = tfidf_model.predict(tfidf_x_test)

# Calculate the TF-IDF model's accuracy.

tfidf_accuracy = accuracy_score(
    tfidf_y_test,
    tfidf_predictions
)

print("\nTF-IDF Accuracy:")
print(tfidf_accuracy)

# Example result:
# 0.3333333333333333

# Print the TF-IDF classification report.

print("\nTF-IDF Classification Report:")
print(
    classification_report(
        tfidf_y_test,
        tfidf_predictions,
        zero_division=0
    )
)