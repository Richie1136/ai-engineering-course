from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch


# Hugging Face integrates seamlessly with popular deep learning frameworks
# such as PyTorch and TensorFlow.

# This allows us to use pre-trained language models within existing machine
# learning workflows, perform inference, and fine-tune models for our own
# tasks.

# Although the Transformers pipeline makes this process simple, working
# directly with the tokenizer and model helps us better understand what is
# happening behind the scenes.


# =====================================================
# Load the Tokenizer
# =====================================================

sentence = "I'm so excited to be learning about large language models."

# print(sentence)

model = "xlnet-base-cased"

# Load the tokenizer for the XLNet model.

tokenizer = AutoTokenizer.from_pretrained(model)

# Tokenize the sentence.

input_ids = tokenizer(sentence)

# print(input_ids)

# The tokenizer returns the tokenized input in a format that the model
# understands.


# =====================================================
# Prepare Inputs for PyTorch
# =====================================================

# To use the model with PyTorch, we need the tokenizer to return tensors
# instead of standard Python objects.

# We'll use a pre-trained sentiment analysis model from Hugging Face.

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# return_tensors="pt" tells the tokenizer to return PyTorch tensors.

input_ids_pt = tokenizer(
    sentence,
    return_tensors="pt"
)

# print(input_ids_pt)

# {
#     'input_ids': tensor([[101, 1045, 1005, 1049, 2061, 7568, 2000,
#                           2022, 4083, 2055, 2312, 2653, 4275,
#                           1012, 102]]),
#
#     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1,
#                                1, 1, 1, 1, 1, 1, 1, 1]])
# }


# =====================================================
# Load the Model
# =====================================================

# Load a pre-trained sequence classification model.

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)


# =====================================================
# Run Inference with PyTorch
# =====================================================

# torch.no_grad() disables gradient calculations because we are only making
# predictions. This improves performance and reduces memory usage.

with torch.no_grad():

    # Feed the tokenized input into the neural network.

    logits = model(**input_ids_pt).logits

# logits contain the raw prediction scores for each possible class.


# =====================================================
# Determine the Predicted Class
# =====================================================

# argmax() returns the index of the largest score.

predicted_class_id = logits.argmax().item()

# Convert the predicted class ID into a readable label.

predicted_label = model.config.id2label[predicted_class_id]

# print(predicted_label)

# POSITIVE


# =====================================================
# Summary
# =====================================================

# These are essentially the same steps performed internally when using the
# Transformers pipeline.

# Working directly with the tokenizer and model gives us greater control over
# the prediction process and helps us understand what happens behind the
# scenes.

# This approach also makes it easier to integrate Hugging Face models into
# PyTorch and TensorFlow applications and provides greater flexibility when
# fine-tuning models for custom tasks.