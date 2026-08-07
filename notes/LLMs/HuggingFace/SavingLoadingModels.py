from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# Hugging Face makes it easy to save and reload models and tokenizers.

# This is useful after fine-tuning a model because you can save it once and
# reuse it later without having to retrain it.


# =====================================================
# Specify the Save Directory
# =====================================================

# Choose where the tokenizer and model will be saved.

model_directory = "my_saved_models"


# =====================================================
# Load a Pre-trained Model and Tokenizer
# =====================================================

model_name = "xlnet-base-cased"

# Load the tokenizer.

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model.

model = AutoModelForSequenceClassification.from_pretrained(model_name)


# =====================================================
# Save the Tokenizer and Model
# =====================================================

# save_pretrained() saves everything needed to reload the tokenizer and model
# later.

tokenizer.save_pretrained(model_directory)

model.save_pretrained(model_directory)


# =====================================================
# Reload the Tokenizer and Model
# =====================================================

# Once the files have been saved, they can be loaded again by specifying the
# directory where they were stored.

my_tokenizer = AutoTokenizer.from_pretrained(model_directory)

my_model = AutoModelForSequenceClassification.from_pretrained(
    model_directory
)


# =====================================================
# Summary
# =====================================================

# Saving models allows us to reuse pre-trained or fine-tuned models without
# downloading or training them again.

# This is especially useful when deploying machine learning applications or
# continuing work on a model at a later time.