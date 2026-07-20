The pandas library explanation
Pandas is one of the most important libraries in Python for data analysis.
It helps you store, clean, explore, and analyze data quickly and efficiently.
You’ll often see it used in data science, machine learning, and general data processing tasks.

Let’s start by understanding what Pandas actually does.

Imagine you have a spreadsheet full of sales data — with columns like Product Name, Price, and Quantity Sold.
Pandas allows you to work with this data directly inside Python, just like you would in Excel — but with much more power and flexibility.

You can use Pandas to:

Load data from files like CSV, Excel, or SQL databases

Filter rows and select specific columns

Perform calculations or summaries (like averages or totals)

Handle missing values

Combine multiple datasets together

In short — Pandas makes data manipulation easy and efficient.



Importing Pandas

Before using Pandas, you first need to import it into your Python script.
This is done with the following line:

import pandas as pd

import pandas tells Python to load the Pandas library so you can use it.

as pd is a shortcut — it allows you to refer to Pandas simply as “pd” in your code.

This alias is used almost universally in Python projects, so you’ll see it everywhere.
For example, instead of writing pandas.DataFrame, you’ll write pd.DataFrame. These two items are the main structures in pandas.

1. The "pd.Series" Object

A Series is a one-dimensional labeled array — like a single column in a spreadsheet.

Here’s how to create one:

import pandas as pd
numbers = pd.Series([10, 20, 30, 40])
print(numbers)
Output:

0    10
1    20
2    30
3    40
dtype: int64
Let’s unpack that:

Each value (10, 20, 30, 40) has a label, called an index (0, 1, 2, 3 by default).

The “dtype” at the bottom shows the data type of the elements (in this case, integers).

You can think of a Series as a column of data with labels.



2. The "pd.DataFrame" Object

A DataFrame is a two-dimensional table — like a full spreadsheet with rows and columns.

Here’s an example:

import pandas as pd
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["London", "Paris", "Berlin"]
}
df = pd.DataFrame(data)
print(df)
Output:

      Name  Age    City
0    Alice   25  London
1      Bob   30   Paris
2  Charlie   35  Berlin
Explanation:

Each key in the dictionary ("Name", "Age", "City") becomes a column.

Each list inside the dictionary becomes the data for that column.

Pandas automatically adds an index column on the left (0, 1, 2).

You can think of a DataFrame as a collection of Series that share the same index.



Syntax Explanation

You’ll often see code written like:

pd.Series()
pd.DataFrame()
This syntax means:

Use the Series or DataFrame class from the Pandas library, which we imported as pd.

The dot (.) tells Python to access something inside the library — in this case, a function or class.

So "pd dot Series" and "pd dot DataFrame" are just the Pandas way of creating structured data containers.



Additional notes: Text preprocessing with pandas and NLTK
When preparing text for analysis or machine learning, we often perform several cleaning and transformation steps to make the text more uniform and easier to work with. Each step addresses a specific issue in raw text — from removing stopwords and punctuation to turning words into tokens and root forms.

Let’s start by analyzing one of these operations in detail and then move through the rest more briefly.

1. Removing Stopwords.

 data['review_no_stopwords'] = data['review_lowercase'].apply(lambda x: ' '.join([word for word \
                                in x.split() if word \
                                not in (en_stopwords)]))
At first, it might look intimidating — but this codeblock is simply doing several small, logical steps one after another. It’s written in one line to be compact, but let’s break it down and explain why each part is there and what it does.

(Note: In Python, a backslash at the end of a line means: “This line of code continues on the next line.” If your line of code is too long, or you want to break it for readability, you can put a \ at the end — and Python will treat the next line as if it’s part of the same command.)

Step 1. Creating a new column

data['review_no_stopwords'] = ...

Here, a new column is being created in the DataFrame called data. In the brackets, we specify the name of that new column, in this case - ['review_no_stopwords']

Everything on the right-hand side will be the result stored in this column. This lets us keep the processed version of the text without overwriting the original one.

Step 2. Selecting the column to transform

data['review_lowercase']

This refers to the existing column that contains the lowercased text we want to process. We start from this column and apply a transformation to each of its entries. As we store the result in a new column data['review_no_stopwords']we make sure we don't modify the content inside ['review_lowercase'].

Step 3. Applying a function to each value

.apply(lambda x: ...)

The "apply" function in pandas runs a given function on every element in a column. Here, that means it processes each text entry one by one.

The "lambda x:" part defines an anonymous function — a short, inline function that doesn’t need a name after it has been used once.
Each x represents a single text value from the column. Here, we name it x because it keeps the code compact but you can replace it with any valid variable name — for example text, review, or even row — and the code will work exactly the same way. x is just a variable name for each value that "apply" sends into the "lambda".

Using a lambda function allows us to describe the transformation directly inside "apply", without defining a separate function elsewhere.

Step 4. Inside the lambda function

This part defines what exactly we want to do to each text entry:

' '.join([word for word in x.split() if word not in (en_stopwords)])

Let’s go through it from the inside out.

1. Splitting the text

x.split()

This turns the text string into a list of individual words. We do this because we can’t check or remove stopwords while the text is one long string — we need to work with separate words. Splitting lets us inspect and filter words individually.

2. The list comprehension

[word for word in x.split() if word not in (en_stopwords)]

This is the heart of the operation.

It builds a new list of words, but only keeps those that are not in the en_stopwords list.

Let’s analyze the structure piece by piece:

word for word in x.split()
This means: go through each word from the list created by x.split(). It’s a loop written in a compact form.

Here, "word" is the loop variable — the name given to each individual element in x.split() as the loop goes through them one by one. But any other name would work equally well.

The only rule: it must be the same name everywhere inside that comprehension — before and after the for.

if word not in (en_stopwords)
This is a filtering condition. It checks every word and only keeps it if it’s not found in the collection of stopwords.
This condition is evaluated for each word one by one.

A list comprehension is just a shortened version of a regular for loop.

If we wrote it in full, it would look like this:

new_list = []
for word in x.split():
    if word not in en_stopwords:
        new_list.append(word)
So, it reads naturally like this in plain English:

“Create a new list of word
for every word in x.split()
if that word is not in en_stopwords.”
The list comprehension is essentially performing two actions at once:

Looping through every word in the text.

Selecting only those that meet the condition (not being a stopword).

The reason we use a list comprehension here is efficiency — it’s faster and more concise than using a traditional for loop with append().
It keeps the logic for looping and filtering in a single readable structure.

3. Joining the filtered words back

' '.join([...])

After filtering, we have a list of the remaining words.
But the column expects text, not a list.
So we join the words back together into one string, with spaces between them.

We use ' '.join() because:

The ' ' before "join" defines what goes between the words — in this case, a single space.

"join" is called on that space string, and the list of words is passed in as an argument.

The result is a single text string, rebuilt from the filtered words.

Step 5. Why this works in one line

Each method and function in this line passes its result directly into the next one.
Here’s the logical chain:

Select the text column.

Apply a function to each entry.

Inside that function:

Split the text into words.

Filter out stopwords using a list comprehension.

Join the remaining words back into a string.

Store the final cleaned text in a new column.

Because every step produces a return value that can be immediately used by the next,
the entire process can be written as a single continuous expression.

In practice, this one-liner is compact but completely logical —
each part is connected by a clear flow of transformation.

Step 6. Key takeaways

"apply" lets you process each value in a column individually.

"lambda" creates a short inline function when defining a separate one isn’t necessary.

"split" prepares the text for word-level processing.

The list comprehension both loops and filters efficiently in one step.

"join" puts the filtered words back together into readable text.

The entire line works as a chain of transformations, passing results from one step to the next.

Continuing the Preprocessing Pipeline

Now that you understand the structure and logic behind this one-line transformation,
you’ll notice that the following text preprocessing steps use a very similar pattern.
Let’s briefly go through each of them.

2. Replacing a Specific Character (the Asterisk)

data['review_no_stopwords_no_punct'] = data \
                                    .apply(lambda x: \
                                    re.sub(r"[*]", \
                                    "star", \
                                    x['review_no_stopwords'] \
                                    ), axis=1)
Here, "re dot sub" replaces all asterisks (*) in the text with the word "star".
We do this to make text more readable and consistent, since product ratings often use asterisks (for example, “5*”).

The pattern r"[*]" simply tells Python to look for the * character.
Using "axis=1" means we’re applying the function row by row, because each row is treated as a small dictionary of columns.

3. Removing All Remaining Punctuation

data['review_no_stopwords_no_punct'] = data. \
                                    apply(lambda x: \
                                    re.sub(r"([^\w\s])", \
                                    "", \
                                    x['review_no_stopwords_no_punct'] \
                                    ), axis=1)
This line removes every punctuation symbol left in the text.
The regular expression r"([^\w\s])" matches anything that’s not a word character (\w) or whitespace (\s).
In other words, it deletes commas, periods, question marks, and similar symbols.

Removing punctuation helps standardize text before analysis and avoids treating punctuation as separate tokens later.

4. Tokenization

data['tokenized'] = data.apply(lambda x: \
                               word_tokenize( \
                               x['review_no_stopwords_no_punct'] \
                               ), axis=1)
Now we convert each cleaned text string into a list of individual tokens (usually words).
The "word_tokenize" function from NLTK is more advanced than "split()" —
it recognizes punctuation and spacing correctly, even in complex text.

Tokenization prepares the data for any operation that needs to process each word separately,
such as stemming or lemmatization.

5. Stemming

ps = PorterStemmer()
data["stemmed"] = data["tokenized"] \
                  .apply(lambda tokens: \
                  [ps.stem(token) \
                   for token in tokens])
Here we initialize a "PorterStemmer" object and apply its "stem" method to every token.
Stemming reduces words to their base form by trimming common endings.
For instance, "running", "runs", and "ran" might all become "run".

We use a list comprehension again because we’re looping through each token and transforming it in one compact expression.

6. Lemmatization

lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
data["lemmatized"] = data["tokenized"] \
                    .apply(lambda tokens: \
                    [lemmatizer.lemmatize(token) \
                     for token in tokens])
Lemmatization looks similar to stemming but is smarter —
it uses a vocabulary and grammatical rules to find the true base form (lemma) of a word.
We use "WordNetLemmatizer" from NLTK, and downloading "wordnet" ensures the lemmatizer has the necessary database.

7. Creating N-grams

tokens_clean = sum(data['lemmatized'], [])
unigrams = (pd.Series \
            (nltk.ngrams(tokens_clean, 1)) \
            .value_counts()) 
print(unigrams)
Finally, we move beyond cleaning and start analyzing patterns in text.

sum(data['lemmatized'], []) flattens the list of token lists into one long list.

nltk.ngrams(tokens_clean, 1) generates unigrams, or single-word combinations.

Wrapping that in pd.Series(...).value_counts() counts how often each unigram appears.

If you change 1 to 2 or 3, you’ll get bigrams or trigrams — sequences of two or three words,
helping identify common phrases or expressions in the text.



Summary

Throughout these steps, we’ve built a compact and powerful text preprocessing pipeline.
Every transformation follows the same logic:
take an existing column, apply a function to transform each text, and store the result in a new column.

Each step — from removing unwanted words to identifying frequent word patterns —
brings the raw text closer to a structured form that algorithms can understand and learn from.