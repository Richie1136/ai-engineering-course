# PartsOfSpeech.py

# 2 types of tagging - Parts of speech tagging and named entity recognition.

# The first method is parts of speech tagging. This is where we take each of our tokens
# and tag them with the associated part of speech.

# By parts of speech, we mean whether that token is a verb, a noun, an adjective, etc.

# I         want        an          early
# Pron      Verb        Det         Adj

# The second method is named entity recognition. Instead of going through each of the tokens
# and tagging them, this method searches through our text and pulls out named entities.

# This means things like people, places, organizations, works of art, or any kind of named
# entity that's easily recognizable by most people.

# These methods of tagging our text can be really useful for exploring our text and understanding
# what's in there. They can also be used to create additional features for machine learning
# algorithms, or they can be an interesting standalone analysis in themselves.

# Sentence

# Barack Obama was born in Honolulu, Hawaii and served as the 44th President of the United States.

# Part of speech tagging is the process of labeling each word in a sentence with its grammatical role.
# For example, whether it's a noun, a verb, or an adjective.

# This helps us understand not only the meaning of the separate words, but how they function in a sentence.

# spaCy - An NLP library that comes with pretrained models capable of recognizing parts of speech,
# named entities, and other linguistic features right out of the box.

# pandas - A library that is great for storing and working with structured data.

import spacy
import pandas as pd  # Pandas will help us organize the tagged text into a clear format so we can analyze it more easily.

nlp = spacy.load('en_core_web_sm')  # Specify the model we want to use. This English core model is lightweight and fast,
# which makes it perfect for learning and smaller projects where speed matters more than perfect accuracy.
# By loading it into the nlp variable, we now have a ready-to-use tool for processing English text.

emma_ja = "emma woodhouse handsome clever and rich with a comfortable home and happy disposition seemed to unite some of the best blessings of existence and had lived nearly twentyone years in the world with very little to distress or vex her she was the youngest of the two daughters of a most affectionate indulgent father and had in consequence of her sisters marriage been mistress of his house from a very early period her mother had died too long ago for her to have more than an indistinct remembrance of her caresses and her place had been supplied by an excellent woman as governess who had fallen little short of a mother in affection sixteen years had miss taylor been in mr woodhouses family less as a governess than a friend very fond of both daughters but particularly of emma between them it was more the intimacy of sisters even before miss taylor had ceased to hold the nominal office of governess the mildness of her temper had hardly allowed her to impose any restraint and the shadow of authority being now long passed away they had been living together as friend and friend very mutually attached and emma doing just what she liked highly esteeming miss taylors judgment but directed chiefly by her own"

# ^ This is often used in natural language processing because it's freely available as public domain text.

# For simplicity, the text has already been preprocessed a bit. Punctuation has been removed,
# and all words have been converted to lowercase.

# However, we've kept the stop words since part of speech tagging also applies to those common
# words like the, and, and is.

# A spaCy document is an object that stores the text along with all the linguistic information
# spaCy generates.

spacy_doc = nlp(emma_ja)  # This single line of code takes the text from Emma and turns it into a structured document,
# where each word is already tokenized and tagged.

pos_df = pd.DataFrame(columns=['token', 'pos_tag'])  # We tell pandas that this table should have two columns,
# one named token and one named pos_tag.

# For each token, we grab two pieces of information: the word itself and its part of speech tag,
# stored in a dictionary. We then turn that dictionary into a one-row table using
# DataFrame.from_records(). Next, we join this one-row table to our larger table, pos_df,
# using the pandas concat method.

# This process happens over and over once for each token in the text. By the time the loop finishes,
# pos_df has grown into a full DataFrame with one row for each token and its part of speech tag.
# Once our table is ready, we can take a quick look at it by running pos_df.head(15).

for token in spacy_doc:
    pos_df = pd.concat([
        pos_df,
        pd.DataFrame.from_records([{'token': token.text, 'pos_tag': token.pos_}])
    ], ignore_index=True)

# print(pos_df.head(15))  # This displays the first 15 words of the text along with the part of speech tags.

# Find the most common tokens and their associated POS tags.

# Inside the table, each unique token and its tag will be grouped, and we will also count how many
# times they appear.

# First, we use the groupby function on pos_df, grouping by both the token and the part of speech tag.

# Then we use .size() to count how many rows belong to each group.

# Then we use reset_index(). This is important because after grouping, pandas uses the grouped columns
# as the index of the table.

# Resetting the index turns them back into regular columns, so our table looks clean and easy to read.

# Finally, we use sort_values() with ascending set to False, which makes sure the most significant
# counts appear first.

pos_df_counts = pos_df.groupby(['token', 'pos_tag']).size().reset_index(name='counts').sort_values(
    by='counts',
    ascending=False
)

pos_df_counts.head(10)  # We see the ten most common tokens along with their POS tags and how many times they appear.
print(pos_df_counts)

# How many different words fall under each POS tag?

# Each row will represent one part of speech tag, such as noun, verb, or adjective,
# along with the number of unique tokens that belong to it.

# We use the groupby function on pos_df_counts, this time grouping by the pos_tag column.

# Then, for each group, we count the number of tokens using .count().

# This gives us the number of different words for each part of speech.

# Next, we use sort_values() with ascending set to False. This puts the largest groups first,
# so we can immediately see which part of speech tags are most common in the text.

pos_df_poscounts = pos_df_counts.groupby(['pos_tag'])['token'].count().sort_values(ascending=False)

print(pos_df_poscounts.head(10))  # We see the 10 most common part of speech tags and their counts.

# To look specifically at the most common nouns in our data, we can filter pos_df_counts
# to include only POS tags that equal NOUN.

# From there, we select the top ten nouns by their counts.

nouns = pos_df_counts[pos_df_counts.pos_tag == 'NOUN'][:10]
print(nouns)