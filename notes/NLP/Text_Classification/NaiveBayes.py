import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# =====================================================
# Create the Dataset
# =====================================================

# Create a pandas DataFrame with two columns:
#
# - text: the sentence that the model will analyze
# - sentiment: the label assigned to the sentence, either positive or negative

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

data = data.sample(
    frac=1,
    random_state=7
).reset_index(drop=True)


# =====================================================
# Separate the Features and Labels
# =====================================================

# Prepare the inputs for our algorithm.

# x holds the text data, which contains the features the model will use.

# y holds the target labels from the sentiment column.

x = data["text"]
y = data["sentiment"]


# =====================================================
# Create the Bag of Words Representation
# =====================================================

# The next step is text vectorization. This means converting our sentences
# into numbers so that the machine learning model can work with them.

# We'll use the Bag of Words approach with Scikit-learn's CountVectorizer
# class.

# Each unique word in the dataset becomes a column.

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


# =====================================================
# Split the Dataset
# =====================================================

# Split the data into training and testing sets.

# The training data is used to teach the model, while the testing data is
# used to evaluate how well the model performs on sentences it has not seen.

# test_size=0.3 means that 30% of the dataset is used for testing and 70% is
# used for training.

# random_state=7 makes the split reproducible.

# stratify=y helps preserve the balance of positive and negative examples in
# both the training and testing sets.

x_train, x_test, y_train, y_test = train_test_split(
    bag_of_words,
    y,
    test_size=0.3,
    random_state=7,
    stratify=y
)


# =====================================================
# Train the Naive Bayes Model
# =====================================================

# Naive Bayes is a simple classification algorithm that works using
# probabilities.

# For example, if we want to separate positive reviews from negative reviews,
# Naive Bayes looks at how often certain words appear in each group.

# It then uses these word counts to estimate the probability that a new review
# belongs to each group. Each word acts like a clue.

# Words such as "great" or "amazing" point toward the positive class, while
# words such as "boring" or "terrible" point toward the negative class.

# Naive Bayes collects these clues, combines them mathematically, and predicts
# the most likely class for a new piece of text.

# It is called "naive" because the algorithm assumes that each word in a
# review is independent of the other words.

# This method works surprisingly well for many text-related tasks, such as
# spam detection and sentiment analysis.

# Create the Multinomial Naive Bayes model.

naive_bayes_model = MultinomialNB()

# Train the model by calling fit() with the training features and labels.

naive_bayes_model.fit(
    x_train,
    y_train
)


# =====================================================
# Evaluate the Naive Bayes Model
# =====================================================

# After training, use predict() to generate sentiment predictions for the
# testing dataset.

naive_bayes_predictions = naive_bayes_model.predict(x_test)

# Compute the accuracy score by comparing the true labels with the labels
# predicted by the model.

naive_bayes_accuracy = accuracy_score(
    y_test,
    naive_bayes_predictions
)

print("Naive Bayes Accuracy:")
print(naive_bayes_accuracy)

# Example result:
# 0.3333333333333333

# The accuracy has improved slightly, but the model still has not performed
# very well.

# The Naive Bayes model shows some improvement because it can classify both
# positive and negative classes. However, there is still room for improvement.