from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch


# =====================================================
# FAQ Chatbot
# =====================================================

# Imagine you're working for a car company, Sunset Motors, that wants to add
# a chatbot to its website to answer some of its customers' most frequently
# asked questions.

# They have asked you to create a prototype of the FAQ chatbot.

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

model = BertForQuestionAnswering.from_pretrained(model_name)

tokenizer = BertTokenizer.from_pretrained(model_name)


# =====================================================
# Sunset Motors Context
# =====================================================

sunset_motors_context = (
    "Sunset Motors is a renowned automobile dealership that has been a "
    "cornerstone of the automotive industry since its establishment in 1978. "
    "Located in the picturesque town of Crestwood, nestled in the heart of "
    "California's scenic Central Valley, Sunset Motors has built a reputation "
    "for excellence, reliability, and customer satisfaction over the past four "
    "decades. Founded by visionary entrepreneur Robert Anderson, Sunset Motors "
    "began as a humble, family-owned business with a small lot of used cars. "
    "However, under Anderson's leadership and commitment to quality, it quickly "
    "evolved into a thriving dealership offering a wide range of vehicles from "
    "various manufacturers. Today, the dealership spans over 10 acres, showcasing "
    "a vast inventory of new and pre-owned cars, trucks, SUVs, and luxury "
    "vehicles. One of Sunset Motors' standout features is its dedication to "
    "sustainability. In 2010, the dealership made a landmark decision to "
    "incorporate environmentally friendly practices, including solar panels to "
    "power the facility, energy-efficient lighting, and a comprehensive recycling "
    "program. This commitment to eco-consciousness has earned Sunset Motors "
    "recognition as an industry leader in sustainable automotive retail. Sunset "
    "Motors proudly offers a diverse range of vehicles, including popular brands "
    "like Ford, Toyota, Honda, Chevrolet, and BMW, catering to a wide spectrum of "
    "tastes and preferences. In addition to its outstanding vehicle selection, "
    "Sunset Motors offers flexible financing options, allowing customers to "
    "secure affordable loans and leases with competitive interest rates."
)


# =====================================================
# FAQ Bot Function
# =====================================================

# Create an FAQ bot function where we can provide a question.

# The model will then look at the context we've provided and return the
# correct response.

# This will be an excellent prototype for our FAQ chatbot for the website.

def faq_bot(question):
    context = sunset_motors_context

    # Encode the question and the context.

    input_ids = tokenizer.encode(question, context)

    # Take our tokens and convert the IDs back to their token values.

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Create our segment embeddings.

    # First, find the separator token within our list of input IDs.

    # From the input IDs, we're looking for the tokenizer's separator token ID.

    sep_idx = input_ids.index(tokenizer.sep_token_id)

    # Use this to compute the number of tokens in segment A and segment B.

    num_seg_a = sep_idx + 1

    # For segment B, take the length of the input IDs and subtract the number
    # of tokens in segment A.

    num_seg_b = len(input_ids) - num_seg_a

    # Create the list of segment IDs.

    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # Feed the data into our model.

    # Convert our input IDs into a PyTorch tensor.

    output = model(
        torch.tensor([input_ids]),
        token_type_ids=torch.tensor([segment_ids])
    )

    # Create the answer start and answer end positions.

    # From the output, use torch.argmax() to find where our answer starts and
    # where our answer ends.

    answer_start = torch.argmax(output.start_logits)

    answer_end = torch.argmax(output.end_logits)

    # If the answer end comes after the answer start, we can create our answer.

    # Our answer is made up of the tokens between the start position and the
    # end position, plus one to include the final token.

    if answer_end >= answer_start:
        answer = " ".join(
            tokens[answer_start:answer_end + 1]
        )

    # If the end token comes before the start token, print a message.

    else:
        print(
            "I'm unable to find the answer to this question, "
            "can you please ask another one?"
        )

    corrected_answer = ""

    for word in answer.split():

        # If the word contains hash values, remove them.

        if word[0:2] == "##":
            corrected_answer += word[2:]
        else:
            corrected_answer += " " + word

    return corrected_answer


# print(faq_bot("Where is the dealership located?"))
# crestwood

# print(faq_bot("What make of cars are available?"))
# ford , toyota , honda , chevrolet , and bmw

# print(faq_bot("How large is the dealership?"))
# 10 acres

# Our model is able to reference the context we gave it to answer these
# questions.