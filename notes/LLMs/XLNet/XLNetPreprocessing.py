import pandas as pd  # Used for working with and organizing our datasets
import numpy as np  # Used for numerical operations and finding the highest prediction values
from cleantext import clean  # Used for cleaning text, including removing emojis
import re  # Used for regular expressions when cleaning text
from transformers import (
    XLNetTokenizer,  # Tokenizes text into the format expected by XLNet
    XLNetForSequenceClassification,  # Loads XLNet for text classification
    TrainingArguments,  # Stores the settings used when training the model
    Trainer,  # Handles the training and evaluation process
    pipeline  # Creates a simple pipeline for using our fine-tuned model
)
from sklearn.model_selection import train_test_split  # Splits data into training, testing, and validation sets
from sklearn.preprocessing import LabelEncoder  # Converts text labels into integer labels
import datasets  # Used to create Hugging Face datasets and DatasetDict objects
import evaluate  # Used to evaluate the model's performance
import random  # Used to select a random example from the validation dataset
import matplotlib.pyplot as plt  # Used to create plots and visualize the label distribution


# =====================================================
# Fine-Tuning XLNet
# =====================================================

# We're going to see how we can fine-tune a model based on our own dataset.

# Our goal is to create a model that can take a piece of text as input,
# determine the emotion of the text, and return the emotion label.

# To do this, we'll be fine-tuning an XLNet model.

# We have a labeled dataset containing a number of different sentences and
# their associated emotion labels.

# We'll provide this data to the model so that it can learn the associations
# between the text and the emotion labels.

# We'll then have our own custom fine-tuned language model.


# =====================================================
# Load the Data
# =====================================================

data_train = pd.read_csv('../../data/emotion-labels-train.csv')

data_test = pd.read_csv('../../data/emotion-labels-test.csv')

data_val = pd.read_csv('../../data/emotion-labels-val.csv')

# print(data_train.head())

# We have a piece of text and an associated label.

#                                                 text label
# 0  Just got back from seeing @GaryDelaney in Burs...   joy
# 1  Oh dear an evening of absolute hilarity I don'...   joy
# 2  Been waiting all week for this game ❤️❤️❤️ #ch...   joy
# 3  @gardiner_love : Thank you so much, Gloria! Yo...   joy
# 4  I feel so blessed to work with the family that...   joy


# =====================================================
# Combine the Datasets
# =====================================================

# Concatenate the training, testing, and validation sets together for now so
# that we can clean them all at once.

data = pd.concat(
    [data_train, data_test, data_val],
    ignore_index=True
)


# =====================================================
# Clean the Text
# =====================================================

# We want to remove emojis from the text.

# Create a new column called text_clean.

# Apply the clean() function to every piece of text in our dataset.

data['text_clean'] = data['text'].apply(
    lambda x: clean(x, no_emoji=True)
)

# Remove @mentions from the text.

data['text_clean'] = data['text_clean'].apply(
    lambda x: re.sub(r"@[^\s]+", "", x)
)

# print(data.head(20))

#                                                  text label                                         text_clean
# 0   Just got back from seeing @GaryDelaney in Burs...   joy  just got back from seeing  in burslem. amazing...
# 1   Oh dear an evening of absolute hilarity I don'...   joy  oh dear an evening of absolute hilarity i don'...
# 2   Been waiting all week for this game ❤️❤️❤️ #ch...   joy  been waiting all week for this game #cheer #fr...
# 3   @gardiner_love : Thank you so much, Gloria! Yo...   joy   : thank you so much, gloria! you're so sweet,...
# 4   I feel so blessed to work with the family that...   joy  i feel so blessed to work with the family that...
# 5   Today I reached 1000 subscribers on YT!! , #go...   joy  today i reached 1000 subscribers on yt!! , #go...
# 6   @Singaholic121 Good morning, love! Happy first...   joy   good morning, love! happy first day of fall. ...
# 7   #BridgetJonesBaby is the best thing I've seen ...   joy  #bridgetjonesbaby is the best thing i've seen ...
# 8   Just got back from seeing @GaryDelaney in Burs...   joy  just got back from seeing  in burslem. amazing...
# 9   @IndyMN I thought the holidays could not get a...   joy   i thought the holidays could not get any more...
# 10               I'm just still . So happy .\nA blast   joy               i'm just still . so happy .\na blast
# 11                   It's meant to be!! #happy #happy   joy                   it's meant to be!! #happy #happy
# 12               💥⚖️Yeah‼️ PAUL‼️⚖️💥  #glorious #BB18   joy                          yeah paul #glorious #bb18
# 13  My morning started off amazing!! Hopefully the...   joy  my morning started off amazing!! hopefully the...
# 14  😱 @cailamarsai you've had me 😂 😂 the whole tim...   joy   you've had me the whole time watching  after ...
# 15           @iamTinaDatta love you so much #smile 😊😊   joy                            love you so much #smile
# 16  @WyoWiseGuy @LivingVertical however, REI did o...   joy    however, rei did offer me the job today as w...
# 17  2 days until #GoPackGo and 23 days until #GoGi...   joy  2 days until #gopackgo and 23 days until #gogi...
# 18  @TheMandyMoore You are beyond wonderful.  Your...   joy   you are beyond wonderful. your singing prowes...
# 19  @luckiiCHARM_ Luckii, I'm changing in so many ...   joy   luckii, i'm changing in so many ways bc of hi...


# =====================================================
# Check the Label Distribution
# =====================================================

# Print the number of pieces of text associated with each label.

# Take the label column, use value_counts(), and plot the results as a bar
# chart.

# We have four emotion classifications:
# - fear
# - anger
# - joy
# - sadness

# data['label'].value_counts().plot(kind="bar")
# plt.show()

# We can see that this is an unbalanced dataset.

# We have more occurrences of text with the label fear compared to sadness,
# so we want to make sure we have an equal number of pieces of text for each
# label.


# =====================================================
# Balance the Dataset
# =====================================================

g = data.groupby('label')

# We're going to sample based on the number of rows in each label.

# Essentially, this groups the DataFrame by label and finds the group with
# the fewest rows.

# In this case, that would be sadness.

# We then use that minimum value to sample the same number of rows from all
# of the other labels.

data = pd.DataFrame(
    g.apply(
        lambda x: x.sample(g.size().min()).reset_index(drop=True)
    )
)

# Check whether the balancing worked by plotting the labels again.

data['label'].value_counts().plot(kind="bar")

# plt.show()

# We can see that we now have an equal number of rows for each label.


# =====================================================
# Convert Labels to Integers
# =====================================================

# Convert our labels into integers.

# Use LabelEncoder().fit_transform() on our label column.

data['label_int'] = LabelEncoder().fit_transform(data['label'])

NUM_LABELS = 4


# =====================================================
# Create Training, Testing, and Validation Splits
# =====================================================

# The final step in preprocessing our text is creating the training and
# testing data.

# First, we'll create our training split and testing split using
# train_test_split() from Scikit-learn.

train_split, test_split = train_test_split(
    data,
    train_size=0.8
)

# Create the validation set.

# Split our training dataset again and put aside 10% for validation.

train_split, val_split = train_test_split(
    train_split,
    train_size=0.9
)

# print(len(train_split))
# 4414

# print(len(test_split))
# 1227

# print(len(val_split))
# 491

# Most of our data is being used for training, some is being used for testing,
# and we've left a smaller sample for validation.


# =====================================================
# Format the Data
# =====================================================

# Format our data and remove any columns that we aren't going to use.

train_df = pd.DataFrame({
    "label": train_split.label_int.values,
    "text": train_split.text_clean.values
})

test_df = pd.DataFrame({
    "label": test_split.label_int.values,
    "text": test_split.text_clean.values
})


# =====================================================
# Create the Dataset Dictionary
# =====================================================

train_df = datasets.Dataset.from_dict(train_df)

test_df = datasets.Dataset.from_dict(test_df)

dataset_dict = datasets.DatasetDict({
    "train": train_df,
    "test": test_df
})

# print(dataset_dict)

# DatasetDict({
#     train: Dataset({
#         features: ['label', 'text'],
#         num_rows: 4414
#     })
#     test: Dataset({
#         features: ['label', 'text'],
#         num_rows: 1227
#     })
# })

# We now have our dataset dictionary split into train and test sets, both
# containing the features label and text.


# =====================================================
# Create the Tokenized Inputs
# =====================================================

# The first step is to load our tokenizer.

# Initialize the tokenizer.

tokenizer = XLNetTokenizer.from_pretrained(
    "xlnet-base-cased"
)


# =====================================================
# Define the Tokenize Function
# =====================================================

def tokenize_function(examples):

    # Return our tokenizer applied to the example text.

    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=128,
        truncation=True
    )


# Apply the tokenize function to our dataset dictionary.

tokenized_datasets = dataset_dict.map(
    tokenize_function,
    batched=True
)

# print(tokenized_datasets)

# DatasetDict({
#     train: Dataset({
#         features: ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask'],
#         num_rows: 4414
#     })
#     test: Dataset({
#         features: ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask'],
#         num_rows: 1227
#     })
# })

# We can see that we've been given a dataset dictionary containing:
#
# - label
# - text
# - input_ids
# - token_type_ids
# - attention_mask


# =====================================================
# Inspect the Tokenized Data
# =====================================================

# print(tokenized_datasets['train']['text'][0])

# o you who have believed, fear allah and believe in his messenger; he will
# [then] give you a double portion of his mercy...' (quran 57:28)

# print(tokenized_datasets['train']['input_ids'][0])

# [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4553,
#  44, 28, 94, 6837, 4587, 778, 78, 12791, 12791, 17, 150, 569, 17, 530,
#  2120, 29, 43, 26, 23, 126, 192, 4, 3]

# We can see that this sentence has been converted into input IDs and starts
# with a lot of 5s.

# Decode 5 to see what it represents.

# print(tokenizer.decode(5))
# <pad>

# These are the PAD special tokens.

# Because we want to pad our data, these tokens have been added to the start
# of the sentence to ensure that all of our inputs are the same length.


# =====================================================
# Inspect the Token Type IDs
# =====================================================

# print(tokenized_datasets['train']['token_type_ids'][0])

# [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#  3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  0, 0, 0, 0, 0, 0, 2]

# The padded tokens have been given a token type ID of 3.

# Our sentence has been given a token type ID of 0, and one of the special
# tokens has been given a token type ID of 2.

# This helps the model distinguish between the different types of tokens
# within the text.


# =====================================================
# Inspect the Attention Mask
# =====================================================

# print(tokenized_datasets['train']['attention_mask'][0])

# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
#  1, 1, 1, 1, 1, 1, 1, 1, 1]

# We can see that all of the PAD tokens have been given a value of 0 because
# the model doesn't need to pay attention to them.

# The values of 1 represent our sentence and special tokens that the model
# should pay attention to.


# =====================================================
# Create Smaller Training and Evaluation Samples
# =====================================================

# Take a sample of the tokenized dataset for the training demonstration,
# otherwise it will take a while to run.

small_train_dataset = (
    tokenized_datasets["train"]
    .shuffle(seed=42)
    .select(range(100))
)

small_eval_dataset = (
    tokenized_datasets["test"]
    .shuffle(seed=42)
    .select(range(100))
)


# =====================================================
# Fine-Tune the Model
# =====================================================

# Fine-tuning a model is made easier by the Hugging Face package.

# Initialize the model.

model = XLNetForSequenceClassification.from_pretrained(
    'xlnet-base-cased',
    num_labels=NUM_LABELS,
    id2label={
        0: 'anger',
        1: 'fear',
        2: 'joy',
        3: 'sadness'
    }
)


# =====================================================
# Create the Evaluation Metric
# =====================================================

# Using the evaluate package, we can specify which measure of accuracy we want
# the model to evaluate while it is training.

metric = evaluate.load('accuracy')


# Define a function to compute the accuracy.

def compute_metrics(eval_pred):

    # Take the logits and labels.

    logits, labels = eval_pred

    # Use the logits to compute our predictions.

    predictions = np.argmax(
        logits,
        axis=1
    )

    # Compute the accuracy using our predictions and labels.

    return metric.compute(
        predictions=predictions,
        references=labels
    )


# =====================================================
# Set Up the Training Arguments
# =====================================================

# The next thing we need to do before training our model is specify a
# directory to save our training information.

# We can also use TrainingArguments to specify custom hyperparameters.

# Set up our training arguments and specify how we want the model to evaluate
# during training.

training_args = TrainingArguments(
    output_dir="test_trainer",
    eval_strategy="epoch",
    num_train_epochs=3
)


# =====================================================
# Create the Trainer
# =====================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

# Run trainer.train() to train our model using all of the arguments we've set
# up in our Trainer.

# trainer.train()

# We can see that the accuracy is evaluated after each epoch.

# Once this finishes running, we have successfully fine-tuned our XLNet model
# using our custom data.

# {'eval_loss': 1.39378023147583, 'eval_accuracy': 0.29, 'eval_runtime': 3.4546,
#  'eval_samples_per_second': 28.947, 'eval_steps_per_second': 3.763, 'epoch': 1.0}

# {'eval_loss': 1.370064377784729, 'eval_accuracy': 0.32, 'eval_runtime': 2.3369,
#  'eval_samples_per_second': 42.791, 'eval_steps_per_second': 5.563, 'epoch': 2.0}

# {'eval_loss': 1.3560224771499634, 'eval_accuracy': 0.29, 'eval_runtime': 2.395,
#  'eval_samples_per_second': 41.754, 'eval_steps_per_second': 5.428, 'epoch': 3.0}

# {'train_runtime': 55.9717, 'train_samples_per_second': 5.36,
#  'train_steps_per_second': 0.697, 'train_loss': 1.4258990165514824,
#  'epoch': 3.0}


# =====================================================
# Evaluate the Model
# =====================================================

# The last thing to do is evaluate the performance of the model.

# trainer.evaluate()

# This provides metrics such as our loss, accuracy, runtime, and more.

# {'eval_loss': 1.4123218059539795, 'eval_accuracy': 0.27,
#  'eval_runtime': 3.4499, 'eval_samples_per_second': 28.986,
#  'eval_steps_per_second': 3.768, 'epoch': 1.0}

# {'eval_loss': 1.4517468214035034, 'eval_accuracy': 0.21,
#  'eval_runtime': 3.5886, 'eval_samples_per_second': 27.866,
#  'eval_steps_per_second': 3.623, 'epoch': 2.0}

# {'eval_loss': 1.4432765245437622, 'eval_accuracy': 0.21,
#  'eval_runtime': 2.4208, 'eval_samples_per_second': 41.309,
#  'eval_steps_per_second': 5.37, 'epoch': 3.0}

# {'train_runtime': 47.1407, 'train_samples_per_second': 6.364,
#  'train_steps_per_second': 0.827, 'train_loss': 1.4426236274914863,
#  'epoch': 3.0}


# =====================================================
# Save the Fine-Tuned Model
# =====================================================

# Let's say we're happy with the model and now want to save our custom model.

model.save_pretrained("fine_tuned_model")


# =====================================================
# Reload the Fine-Tuned Model
# =====================================================

# Load the model by specifying the fine_tuned_model folder where it was saved.

fine_tuned_model = XLNetForSequenceClassification.from_pretrained(
    'fine_tuned_model'
)


# =====================================================
# Create a Classification Pipeline
# =====================================================

# Create a pipeline using our fine-tuned model and tokenizer.

clf = pipeline(
    "text-classification",
    fine_tuned_model,
    tokenizer=tokenizer
)


# =====================================================
# Test the Fine-Tuned Model
# =====================================================

# Run the pipeline on a sample from our validation dataset.

# First, specify a random number so that we can select a random row from the
# dataset.

rand_int = random.randint(
    0,
    len(val_split)
)

# Print the text from this row.

print(
    val_split['text_clean'][rand_int]
)

# i'd let jiho step on my throat but he'd probs be afraid he'd hurt me

# Use the classification pipeline on our text and specify top_k=None so that
# we can see the prediction scores for each of the labels we've trained.

answer = clf(
    val_split['text_clean'][rand_int],
    top_k=None
)

print(answer)

# [{'label': 'joy', 'score': 0.280955970287323},
#  {'label': 'fear', 'score': 0.27818816900253296},
#  {'label': 'anger', 'score': 0.2382514476776123},
#  {'label': 'sadness', 'score': 0.20260436832904816}]

# We've successfully created a fine-tuned model based on our own data.

# In this course, you've learned a number of different ways to interact with
# large language models.

# Now you're able to successfully fine-tune a model for a specific task using
# your own custom data.