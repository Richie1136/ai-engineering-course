from transformers import (
    RobertaTokenizer,
    RobertaModel,
    DistilBertTokenizer,
    DistilBertModel,
)


# =====================================================
# BERT Variants
# =====================================================

# If you do some further reading into the BERT model, or you look up BERT
# models on the Hugging Face model hub, you'll eventually come across some
# variations of the BERT model.

# These include RoBERTa and DistilBERT.

# BERT, RoBERTa, and DistilBERT are all variations of the original BERT model,
# each with its own unique characteristics and modifications.


# =====================================================
# Key Differences Between the Models
# =====================================================

# RoBERTa

# RoBERTa stands for Robustly Optimized BERT Pre-training Approach.

# It is a modification and optimization of the original BERT model.

# It employs a larger batch size, longer training sequences, and removes the
# Next Sentence Prediction task during pre-training.

# RoBERTa also uses dynamic masking, where the masking pattern is changed for
# each training batch.

# The result is a model that achieves better performance on various downstream
# NLP tasks compared to the original BERT.

# RoBERTa is known for its robustness and is generally considered an
# improvement over BERT.


# DistilBERT

# DistilBERT is a smaller and more lightweight version of BERT, designed for
# efficiency and faster inference.

# It is distilled from the BERT model, where it is trained to mimic the
# behavior of the larger BERT model while using fewer parameters.

# It has 40% fewer parameters than the BERT Base model.

# It runs about 60% faster while preserving over 95% of BERT's performance.

# DistilBERT retains much of BERT's performance on various NLP tasks, but with
# significantly fewer parameters, making it more suitable for
# resource-constrained environments.


# =====================================================
# Load the RoBERTa Model
# =====================================================

rob_model_name = "roberta-base"

rob_tokenizer = RobertaTokenizer.from_pretrained(rob_model_name)

rob_model = RobertaModel.from_pretrained(rob_model_name)


# =====================================================
# Load the DistilBERT Model
# =====================================================

dis_model_name = "distilbert-base-uncased"

dis_tokenizer = DistilBertTokenizer.from_pretrained(dis_model_name)

dis_model = DistilBertModel.from_pretrained(dis_model_name)