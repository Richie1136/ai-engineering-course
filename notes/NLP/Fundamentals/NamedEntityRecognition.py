# NamedEntityRecognition.py

# Named Entity Recognition (NER) is the process of finding and labeling essential pieces
# of information in text, such as names of people, places, organizations, dates, or quantities.

# For example, in the sentence "Emma lives in London," NER would tag Emma as a person and
# London as a location. However, this goes beyond simply finding capitalized words.

# NER uses a combination of rules and machine learning to discover entities. While capital
# letters are one clue, the model also looks at the surrounding context, word patterns,
# and statistical probabilities. This way, Apple might be tagged as a fruit in one sentence
# but as a company in another.

# This makes NER a powerful tool for turning unstructured text into structured information
# that can be searched, analyzed, or used in applications like chatbots, question answering,
# and recommendation systems.

# spaCy comes with powerful pre-trained models.

# displaCy is a built-in visualization tool that lets you see entities highlighted directly in text.

# tokenizer breaks text into tokens so spaCy can process it correctly.

# Finally, we bring in HTML and display from IPython, which lets us render visualizations.

import spacy
from spacy import displacy
from spacy import tokenizer
from IPython.display import HTML, display
import re


# spaCy model - The engine that will process the text for named entity recognition.

nlp = spacy.load('en_core_web_sm')  # Small English model.

# The text we will use is taken from the Google Wikipedia page.

google_text = """Google was founded on September 4, 1998, by computer scientists Larry Page and Sergey Brin while they were PhD students at Stanford University in California. Together they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock. 
The company went public via an initial public offering (IPO) in 2004. In 2015, Google was reorganized as a wholly owned subsidiary of Alphabet Inc. Google is Alphabet's largest subsidiary and is a holding company for Alphabet's internet properties and interests. Sundar Pichai was appointed CEO of Google on October 24, 2015, replacing Larry Page, who became the CEO of Alphabet. On December 3, 2019, Pichai also became the CEO of Alphabet."""

# spaCy breaks the text into tokens, analyzes the grammar, and looks for named entities.

spacy_doc = nlp(google_text)

# To see what spaCy found, we need to loop through the entities.
# spaCy stores all the entities it detects in an attribute called spacy_doc.ents.

for word in spacy_doc.ents:
    print(word.text, word.label_)  # Prints the entity text and its human-readable label.

# This tells us which parts of the text are detected as dates, people, organizations,
# and other entity categories.


# Visualize the different entities directly within our text.

# displacy.render() takes our spaCy document and creates an HTML version of it
# with the entities highlighted.

# 3 Arguments:

# Arg 1 - The document with the text that spaCy has already analyzed for entities.
# Arg 2 - style="ent" tells spaCy that we want to visualize entities specifically,
# since displaCy can also visualize other elements.
# Arg 3 - jupyter=False means that instead of trying to display the visualization
# automatically in a Jupyter notebook, it returns it as HTML code. That way, we can
# decide how and where to display it.

html = displacy.render(spacy_doc, style="ent", jupyter=False)

# This highlights entities and assigns each category a distinct color, making people,
# organizations, and locations easy to distinguish.

display(HTML(html))

# All that output is stored in the html variable, so we can wrap it with the HTML class
# to tell Python to treat the string as HTML content, not plain text.

# Finally, we pass it into display to see our text with the entities highlighted
# in different colors.

# displacy.serve(spacy_doc, style="ent", port=5001)  # Serves the visualization so we can see it.

google_text_clean = re.sub(r"[^\w\s]", "", google_text).lower()

# Removes punctuation and converts the text to lowercase.

print(google_text_clean)

# Create a new spaCy document with our cleaned text.

spacy_doc_clean = nlp(google_text_clean)

# Iterate through each entity in our new spaCy document and print each entity
# with its associated entity tag.

for word in spacy_doc_clean.ents:
    print(word.text, word.label_)  # Fewer entities are detected because of the lack of punctuation and capitalization.

html = displacy.render(spacy_doc_clean, style="ent", jupyter=False)
display(HTML(html))

# displacy.serve(spacy_doc_clean, style="ent", port=5002)

# Serves the visualization so we can see it.

# There are fewer entities because when we removed capitalization, the model lost a vital clue
# that helps it recognize names of people, places, or organizations. By removing punctuation,
# we also disrupted the sentence boundaries that depend on it to be recognized correctly.
# That's why it's essential to think carefully about when you perform named entity recognition
# in your data cleaning pipeline.

# As you can see, if you clean the text too aggressively before running NER, for example,
# by removing punctuation or converting everything to lowercase, the model may struggle
# to recognize certain entities. One option is to run NER on the raw text first. Another
# option is to try some light preprocessing, then test NER again. In some cases,
# preprocessing can help the model pick up additional entities.

# The key takeaway is to familiarize yourself with your data, understand its context,
# and experiment with where NER fits best in your pipeline. That way, you avoid losing
# important information during cleaning.