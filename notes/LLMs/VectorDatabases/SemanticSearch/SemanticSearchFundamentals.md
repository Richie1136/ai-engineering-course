# Semantic Search Fundamentals

## 1. What Is Semantic Search?

Semantic search, also known as similarity search, is a search technique that
focuses on the meaning and context of a query rather than relying exclusively
on exact keyword matches.

Instead of searching only for identical words, semantic search identifies
content that is similar in meaning to the user's query.

For example:

- Traditional keyword search looks for exact words or matching terms.
- Semantic search looks for content with similar meanings, even when different
  words or phrases are used.

Semantic search is commonly implemented using embeddings and vector databases.

---

## 2. Example: Searching for a News Article

Imagine you're using a newsreader application.

A few weeks ago, you read an interesting article about the life of
Queen Elizabeth II.

You remember an interesting passage from the article but have forgotten
some of the details.

You decide to search for the article using the application's search bar.

### First Search: Queen Elizabeth Retrospective

You enter:

**Queen Elizabeth Retrospective**

Unfortunately, the article you're looking for doesn't appear.

Instead, the search results contain more recent articles about the
British Royal Family.

These articles mention Queen Elizabeth II in passing but aren't
specifically dedicated to her life.

### Second Search: Searching Headlines

You refine your search to include only article headlines.

You're confident that the information you're looking for appears
somewhere in the title.

Unfortunately, the search returns no results.

### Finding the Article

After scrolling through several pages of search results, you finally
find the article.

Its title is:

**Elizabeth II and the Monarch's Life and Reign**

The article didn't appear in your earlier searches because its title
didn't contain the exact words from your original query.

Additionally, the search results were primarily ordered by recency
rather than relevance.

Because the article was published approximately 30 days earlier,
it appeared several pages into the search results.

This illustrates a limitation of search systems that rely heavily on
exact keyword matching and chronological ordering.

---

## 3. How Does Semantic Search Solve This Problem?

Instead of requiring exact keyword matches, semantic search identifies
content with similar meanings.

For example, consider these terms:

- Queen
- Monarch
- Elizabeth II
- Queen Elizabeth II

Although these terms aren't identical, they are closely related
in the context of the search.

A semantic search system can recognize these relationships through
text embeddings.

### How It Works

1. Text is converted into numerical vectors using an embedding model.
2. These vectors represent semantic information about the text.
3. The vectors are stored in a vector database.
4. A user's search query is converted into an embedding using the same model.
5. The database compares the query vector against stored vectors.
6. Results are ranked according to their similarity to the query.

In our example, searching for:

**Queen Elizabeth Retrospective**

could return:

**Elizabeth II and the Monarch's Life and Reign**

Even though the wording differs, the two phrases share a similar meaning.

The article could therefore receive a high similarity score and appear
near the top of the search results.

### Traditional Search vs. Semantic Search

**Traditional keyword search:**

- Focuses on matching words or phrases.
- Can use exact matches, partial matches, and relevance-ranking techniques.
- May struggle when users search with different terminology.
- Does not necessarily understand the meaning behind a query.

**Semantic search:**

- Focuses on the meaning and context of the query.
- Uses embeddings to represent text numerically.
- Can identify relevant content even when exact keywords differ.
- Often uses vector similarity to rank search results.

**Important:** Traditional search engines are not limited to exact matching.
Many support stemming, synonyms, full-text indexing, and relevance ranking.
Semantic search adds another way to identify relationships between queries
and documents.

---

## 4. Why Are Vector Databases Important?

Vector databases allow us to store and search numerical representations
of data.

These numerical representations are called embeddings.

When text is converted into embeddings, content with similar meanings
will generally have vectors that are close together in the embedding space.

This allows us to perform similarity searches efficiently.

For example:

**Search Query:** Queen Elizabeth Retrospective

**Relevant Article:** Elizabeth II and the Monarch's Life and Reign

The vector database can identify the article as relevant because
the embeddings represent related meanings.

This is particularly useful when working with large collections
of documents.

---

## 5. Traditional Databases vs. Vector Databases

One of the main considerations in this case study is how organizations
store and search their data.

A large amount of business data is stored in relational or
semi-structured formats.

Examples include:

- SQL databases
- NoSQL databases
- Structured tables
- Business records
- Course catalogs

For many businesses, traditional databases are the most appropriate
solution for storing and managing their data.

Storing every type of business data exclusively as vectors
wouldn't make sense.

### When Traditional Databases Are Useful

Traditional databases are especially useful for:

- Managing structured records.
- Performing exact lookups.
- Filtering data using specific conditions.
- Maintaining relationships between records.
- Supporting transactions and business operations.

### When Vector Databases Are Useful

Vector databases are particularly useful for:

- Semantic search.
- Similarity search.
- Finding related documents.
- Searching large collections of unstructured text.
- Retrieving information based on meaning rather than exact wording.

### Combining Both Approaches

Traditional databases and vector databases don't have to replace
one another.

They can work together.

For example, a company might store its course information in a
traditional database while also storing embeddings of course
descriptions in Pinecone.

This allows the company to maintain its structured records while
supporting semantic search.

---

## 6. Our Semantic Search Case Study

In this case study, we'll explore how to transform structured
course data into embeddings and upload them to a vector database.

We'll use Pinecone to perform semantic searches across
educational content.

The goal is to help students find relevant learning materials,
even when their search terms don't exactly match the course titles
or section names.

### Case Study Game Plan

1. Define the problem we're trying to solve.
2. Familiarize ourselves with the case study dataset.
3. Preprocess and prepare the data using Python.
4. Generate embeddings and upload the data to Pinecone.
5. Perform semantic searches across our courses.
6. Refine the dataset and search configuration to improve relevance.

---

## 7. Applying Semantic Search to Educational Content

Semantic search can be particularly useful for educational platforms.

Students often search for topics rather than specific course titles.

For example, a student might search for:

**Unsupervised Learning in Python**

However, the relevant material might be located inside a course
with a different title.

Traditional keyword-based search systems may fail to return
these results if they rely too heavily on exact matches.

Semantic search can help identify relevant educational content
even when the search query and course titles use different terminology.

### Different Levels of Semantic Search

We could implement semantic search at different levels.

**Course-Level Search**

Search across entire courses using information such as:

- Course titles.
- Course descriptions.
- Course topics.

**Section-Level Search**

Search across individual sections within courses.

This is useful when a topic appears in only part of a larger course.

**Combined Course-Level and Section-Level Search**

Search both courses and their individual sections.

This can help students discover relevant content that is spread
across multiple courses.

Depending on the application's requirements, these approaches
could use separate indexes, namespaces, or other organizational
strategies within a vector database.

---

## 8. Example: Searching the 365 Data Science Platform

Let's examine a search problem involving the 365 Data Science platform.

Suppose a student wants to learn about unsupervised learning in Python.

They enter:

**Unsupervised Learning in Python**

Unfortunately, the search returns zero results.

However, the platform contains relevant material in several courses.

For example, the course:

**Customer Analytics in Python**

contains a section covering K-means clustering with PCA.

Both K-means clustering and PCA are commonly associated with
unsupervised learning techniques.

Therefore, this course contains material relevant to the
student's original search.

### Second Search: Clustering in Python

The student tries another query:

**Clustering in Python**

Unfortunately, this search also returns no results.

### Third Search: Clustering

The student simplifies the query to:

**Clustering**

This time, the search returns two courses:

- Machine Learning in Python.
- Machine Learning in Excel.

These courses contain sections related to clustering
and clustering analysis.

However, the Customer Analytics in Python course
still doesn't appear.

---

## 9. Why Does Traditional Search Miss Relevant Courses?

Consider the search:

**Clustering in Python**

The Machine Learning in Python course contains relevant
sections about clustering.

However, the word "Python" appears in the course title,
while the word "clustering" appears in its sections.

If the search system doesn't effectively combine information
from course titles and section titles, it may fail to
recognize the relationship.

As a result, the search can be treated as an incomplete match.

### Another Example: Customer Analytics in Python

The Customer Analytics in Python course contains three
sections related to clustering.

However, these sections don't appear in the search results.

This demonstrates how a search system can overlook relevant
material when it relies too heavily on exact keyword matches
or searches only certain fields.

### How Semantic Search Can Help

Semantic search could identify relationships between:

- Unsupervised learning.
- Clustering.
- K-means clustering.
- Principal Component Analysis (PCA).
- Customer analytics.

For example, searching for:

**Unsupervised Learning in Python**

could return relevant sections from:

**Customer Analytics in Python**

even when the original search phrase doesn't appear
in the course or section titles.

This would help students discover relevant learning materials
more efficiently.

---

## 10. Semantic Search vs. Generative AI

Another possible solution would be to introduce an AI chatbot.

For example, a student could ask:

**Which courses cover clustering in Python?**

A chatbot might generate a response explaining which courses
contain relevant material and why they're useful.

This approach uses generative AI.

### How Generative AI Differs from Semantic Search

**Semantic Search**

- Retrieves existing content.
- Identifies relevant documents using similarity.
- Ranks results according to their relevance.
- Doesn't require generating new descriptions or explanations.

**Generative AI**

- Generates new text in response to a user's question.
- Can summarize and explain retrieved information.
- Can provide conversational answers.
- May produce inaccurate or unsupported information.

### The Risk of Hallucinations

Generative AI models can sometimes produce incorrect information.

These errors are commonly referred to as hallucinations.

For example, a chatbot might recommend a course and claim
that it covers a particular topic when it actually doesn't.

This could mislead students.

Although techniques such as retrieval-augmented generation (RAG)
can help ground chatbot responses in retrieved information,
they don't completely eliminate the possibility of errors.

---

## 11. Why We're Focusing on Semantic Search

For this case study, we'll focus on semantic search rather
than building a generative AI chatbot.

Our goal is to help students find existing educational content.

We want to:

- Retrieve relevant courses.
- Identify useful course sections.
- Improve search relevance.
- Reduce dependence on exact keyword matches.
- Help students discover material across different courses.

Instead of generating new descriptions of the content,
we'll use embeddings and a vector database to retrieve
existing learning materials.

This keeps the application focused on information retrieval
rather than text generation.

---

## 12. Summary

In this lesson, we learned that:

1. Semantic search focuses on meaning rather than relying
   exclusively on exact keyword matches.

2. Embedding models convert text into numerical vectors
   that represent semantic information.

3. Vector databases allow us to compare embeddings
   and retrieve similar content.

4. Traditional databases remain useful for structured data
   and can work alongside vector databases.

5. Semantic search can help students discover relevant
   courses and sections even when their search terms
   don't match the original titles.

6. Course-level and section-level search can be combined
   to improve content discovery.

7. Semantic search retrieves existing information,
   while generative AI can produce new text.

8. Generative AI introduces additional considerations,
   including the possibility of hallucinations.

In the upcoming lessons, we'll prepare our course dataset,
generate embeddings, upload them to Pinecone, and perform
semantic searches.