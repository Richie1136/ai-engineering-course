x4 = "Robert"
print(x4)

y = 30
print(str(y) + " Dollars")

print("I'm Fine")

sentence = "Her cat's name is Luna"
lower_sentence = sentence.lower()
print(lower_sentence)

sentence_list = ['Could you pass me the TV remote?',
                 'It is IMPOSSIBLE to find this hotel',
                 'Want to go for dinner on Tuesday?']

lower_sentence_list = [x.lower() for x in sentence_list]
print(lower_sentence_list)

# Removing Stop words

# "and", "of", "a", "to"

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

en_stopwords = stopwords.words('english')

sentence = "it was too far to go to the shop and he did not want her to walk"
# go through each word in the sentence variable and return the words that are not in the en_stopwords list and join them back together into a string
sentence_no_stopwords = " ".join([word for word in sentence.split() if word not in en_stopwords])
print(sentence_no_stopwords)

en_stopwords.remove('did')
en_stopwords.remove('not')
en_stopwords.append('go')

sentence_no_stopwords_custom = " ".join([word for word in sentence.split(" ") if word not in en_stopwords])
print(sentence_no_stopwords_custom)

# Regular expressions (regex): a special way of writing patterns to 
# search through text.

import re # Python's built-in library for working with regex

# "\n" = new line
# r"\n" = "\n" - R means raw and tells python to treat backslash and special character exactly as they are, instead
# of giving them a special meaning.

my_folder = "C/desktop\notes"
print(my_folder)
my_folder2 = r"C:\desktop\notes"
print(my_folder2)

result_search = re.search("pattern", r"string to contain the pattern")
print(result_search)

result_search2 = re.search("pattern", r"the phrase to find isn't in this string")
print(result_search2)

string = r"sara was able to help me find the items I needed quickly"
new_string = re.sub("sara", "Sarah", string)
print(new_string)

customer_reviews = ["sam was a great help to me in the store",
                    "the cashier was very rude to me, I think her name was eleanor",
                    "amazing work from sadeen!",
                    "sarah was able to help me find the items I needed quickly",
                    "lucy is such a great addition to the team",
                    "great service from sara she found me what I wanted"
                    ]

sarahs_review = []

pattern_to_find = r"sarah?"

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        sarahs_review.append(string)
print(sarahs_review)

a_reviews = []
# ^ means to look at the start of the string.
pattern_to_find = r"^a"

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        a_reviews.append(string)

print(a_reviews)

y_reviews = []

# $ means to look at the end of the string.
pattern_to_find = r"y$"

for string in customer_reviews:
    if re.search(pattern_to_find, string):
        y_reviews.append(string)

print(y_reviews)

# | works like an "or"

needwant_reviews = []

pattern_to_find = r"(need|want)ed"

for string in customer_reviews:
    if re.search(pattern_to_find, string):
        needwant_reviews.append(string)

print(needwant_reviews)

no_punct_reviews = []

# [] - A set of characters we want to match
# ^ - "not"
# \w - word characters
# \s - white space characters
pattern_to_find = r"[^\w\s]"

for string in customer_reviews:
    # Replace any punctuation we find with nothing
    # third argument is the text we want to clean
    no_punct_string = re.sub(pattern_to_find, "", string)
    no_punct_reviews.append(no_punct_string)

print(no_punct_reviews)
