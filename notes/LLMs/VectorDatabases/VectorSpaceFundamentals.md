# Vector Space Fundamentals

## Vector Spaces

As we embark on this journey, think of yourself as an explorer venturing into a new universe—a universe not bound by the familiar three dimensions of our everyday world, but one that extends far beyond into realms that can only be navigated with the compass of mathematics and the map of intuition.

At the heart of every vector database lies the concept of a **vector space**.


## What Is a Vector Space?

Simply put, a vector space is an abstract mathematical structure where vectors reside.

These vectors are not just arrows pointing from one place to another. In the context of vector databases, they're also carriers of information, with each dimension representing a different feature or characteristic of the data.

Imagine walking into a gallery where each painting is not just a flat image, but a portal to a world with many dimensions.

Some dimensions represent:

- Colors
- Brushstrokes
- Emotions

Each piece of data is like one of these paintings in a vector space, described by its unique set of dimensions.

Mathematically, a vector space is a set whose elements, also known as vectors, can be added together or multiplied by numbers.

We can exploit these properties when working with a vector database.


## Vector Spaces and Similarity Search

Vector spaces are the foundation upon which similarity search is built.

When we search for something similar in a vector database, we look for similarity within this multidimensional space.

The closer two vectors are, the more similar the data they represent.

With that in mind, let's explore how this works using a gallery of paintings as an example.


# Painting Example

In this gallery, each painting is not just an image, but a complex combination of various features captured within the multidimensional space of a vector database.

These features can include:

- Painting name
- Epoch or period
- Style
- Technique
- Size
- Current location

Each of these characteristics contributes to defining the position of a painting within the vector space.


## Mona Lisa

For instance, Leonardo da Vinci's *Mona Lisa* is not just a portrait of a woman, but a masterpiece from the Renaissance characterized by specific painting techniques and an emotional depth that resonates through the ages.


## The Lady with an Ermine

Nearby, you might find *The Lady with an Ermine*, another of da Vinci's works.

It shares many features with the *Mona Lisa*, such as:

- The artist's unique style
- The era
- Emotive expression

These shared characteristics position it closer to the *Mona Lisa* within the vector space.


## Girl with a Pearl Earring

Similarly, Johannes Vermeer's *Girl with a Pearl Earring*, while also a portrait of a woman, sits in a slightly different part of the vector space.

It shares the portrait style and emotional qualities, but differences in the artist's technique, the Dutch Golden Age period, lighting, and color palette set it apart.

However, it remains relatively close to da Vinci's work because of the overarching similarities in subject and form.


## Encoding the Paintings

By observing the paintings, we can identify similarities and differences such as:

- Style
- Artist
- Era
- Technique
- Subject
- Color

The key is to capture this information and encode it into numerical vectors.

This could involve straightforward encoding, such as assigning numbers to different artists, or more complex feature extraction, such as analyzing the color spectrum used in a painting.

Once we populate our vector database with these encoded paintings, we can perform powerful similarity searches.


## Similarity Search Examples

For example, a query for Renaissance portraits could surface works by da Vinci, Michelangelo, and Raphael clustered together because of shared features in:

- Style
- Epoch
- Technique

On the other hand, a search for portraits of women could span more widely, pulling in works from different periods and artists that are similar in subject and form.

In this way, vector databases transform how we interact with data, opening new dimensions of discovery and insight in fields ranging from art history to digital media, marketing, and beyond.


# Measuring Similarity

## How Do We Measure Similarity?

Now we'll examine how we measure the closeness and similarity between data points.

We've already discussed that a vector space has certain properties, such as adding, multiplying, and scaling vectors.

Now we'll discuss how we measure the distance between vectors.

Our main topic is **distance metrics**.

These metrics originate from linear algebra but have practical applications when building vector databases.

In the world of vector databases, distance metrics help us determine how close or similar data points are to each other within a vector space.

There are different ways to measure that closeness depending on which metric we use.


# Euclidean Distance

Euclidean distance is probably the most common distance metric.

It measures the straight-line distance between two points in space.

It's similar to using a ruler to measure the distance between two dots on paper in a two-dimensional space.

Mathematically, for points **p** and **q** in an n-dimensional space, Euclidean distance is defined as the square root of the sum of the squared differences between the two points.

This metric is particularly useful for applications where actual physical distance is important, such as:

- Spatial navigation
- GPS
- Clustering


# Manhattan Distance

Another popular metric is **Manhattan distance**, also known as:

- City-block distance
- Taxicab distance

It takes its name from the famous Manhattan grid, where cars or cabs must travel along grid-like streets.

Manhattan distance measures the distance between two points when movement occurs only along orthogonal, right-angle paths.

For points **p** and **q**, it's calculated as the sum of the absolute differences between their Cartesian coordinates.

This distance can be particularly relevant in scenarios such as:

- Urban planning simulations
- Medical imaging
- Game development

For example, you wouldn't be able to drive diagonally through buildings to reach a destination. Instead, you would need to follow the available paths.


# Dot Product

Another metric commonly used with vectors is the **dot product**.

The dot product, also known as the **scalar product** or **inner product**, is a fundamental operation in vector mathematics.

It also plays a significant role in vector databases, particularly in similarity search and machine learning.

The dot product of two vectors combines information about:

- Their magnitudes
- The angle between them

For two vectors **a** and **b**, calculating the dot product produces a single scalar value.

The magnitude, or length, of a vector is calculated using the square root of the sum of each scalar value squared.

The important point is that the dot product relates the magnitude and angle between two vectors.


## What Does the Dot Product Indicate?

A **positive dot product** indicates that the vectors point in roughly the same direction.

A **zero dot product** indicates orthogonality, meaning the vectors are perpendicular.

A **negative dot product** indicates that the vectors point in opposite directions.

Calculating the dot product is also computationally efficient, making it useful for high-volume, high-dimensional data processing.

This efficiency is important for real-time applications such as:

- Search engines
- Recommendation systems

These applications often require quick similarity assessments.

The dot product also leads us to the final similarity concept in this lesson: **cosine similarity**.


# Cosine Similarity

Cosine similarity is closely related to the dot product.

Rather than focusing on the literal distance between two points, cosine similarity evaluates the cosine of the angle between two vectors.

It is calculated using the dot product divided by the magnitudes of the two vectors.

The key idea is that we're interested in the **orientation and direction** of the vectors.


## Same Direction = Same Category

For example, imagine a two-dimensional vector space containing the following words:

- Clever
- Dandelion
- Flower

The vectors representing **dandelion** and **flower** would tend to point in similar directions because they belong to the same general floral category.

Their similarity score would therefore be high.

This approach is less about literal distance and more about orientation and direction.

It's an important metric in areas such as:

- Text analysis
- Recommendation systems

In these applications, the direction or context of vectors can be more important than their magnitude.


## Similarity Search Example

For instance, when a user uploads an image to a search engine to find similar images, the engine can use these metrics to compare the user's image vector with vectors stored in the database.

It can then rapidly identify vectors with the smallest distances or highest similarities.

By understanding these distance and similarity metrics, we can navigate vector spaces more precisely and select approaches based on the needs of an application.

This could include:

- Grouping similar items
- Recommending products
- Identifying patterns

These metrics provide ways to measure how similar vectors are.


# Vector Embeddings

## What Are Vector Embeddings?

Vector embeddings are a pivotal concept in modern:

- Natural Language Processing (NLP)
- Machine Learning (ML)

Embeddings allow us to transform raw data into the structured, machine-understandable format of vectors.

Embeddings are dense representations of data items in a continuous vector space.

These data items could include:

- Words
- Products
- Users

Each vector dimension captures some aspect of the data item, allowing similar items to be represented closer together within the vector space.


## Why Do We Need Embeddings?

We need embeddings because computers don't work directly with the meaning of words the way humans do, but they work very well with numbers.

During the embedding process, we take data such as words and create arrays of numbers representing different characteristics of that data.


# Embedding Example

Consider these two sentences:

> He plays the LEAD guitar in a band.

> They found high levels of LEAD in the drinking water.

These sentences contain the same word, **LEAD**, but that word has a different meaning depending on its context.


## How Would We Embed These Sentences?

For simplicity, let's say each vector has only two dimensions.

In practice, embeddings generally contain many more dimensions.

For this example, let's assume:

- **Dimension 1:** Relevance to musical contexts
- **Dimension 2:** Relevance to substance-related contexts


# First Sentence

> He plays the LEAD guitar in a band.

For this musical context, we'll assign vectors with relatively higher scores in the first dimension and lower scores in the second.


### He

"He" is a pronoun that is mostly irrelevant to either domain.

```text
[0.1, 0.02]