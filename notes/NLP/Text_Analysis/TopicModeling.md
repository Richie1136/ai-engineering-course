# Topic Modeling

## Overview

Topic modeling takes a collection of documents, with documents meaning individual pieces of text. This could be rows in a DataFrame with text in it, items in a list, or anything similar. These are all referred to as documents.

Topic modeling scans each of our documents to identify patterns. We can then group similar documents together into topics.

Topic modeling is an example of **unsupervised learning**. We don't need any labels in our data to find these topics. The algorithm works by identifying similar word patterns throughout different documents. These word patterns are then used to identify what each piece of text is talking about. Once the algorithm understands this, it can group similar documents together.

---

## Example

AI is now able to create videos and audio that are hard to distinguish from the real thing. Recently, a number of celebrities have been impersonated, and journalists need to take extra care verifying sources before going to publication. You don't need any special hardware to work with AI. Governments are discussing bringing in regulations and policy.

When you look through this text, you might see that there are a couple of different themes that stand out.

One topic could be around **equipment**. You have the words **video**, **audio**, and **hardware**, which could be grouped together into one topic.

You then have words like **celebrities**, **journalists**, and **publication**, which could form a second topic around **media**.

You also have the words **government**, **regulations**, and **policy**, which could form a third topic.

Topic modeling looks for key themes within our text. Once it identifies these key themes, it can determine an overall theme for the document and identify similar documents.

---

## Why Topic Modeling Is Useful

Topic modeling is a great resource for any NLP data scientist. It can work quickly and identify key themes and patterns in our data. It can also take a large amount of text and find patterns that humans might miss if we were to do it manually.

There are a number of different algorithms used in topic modeling, but we will be focusing on the two most common.

1. Latent Dirichlet Allocation (LDA)
2. Latent Semantic Analysis (LSA)

---

## Example Topics

* **Topic 1 – Equipment:** Video, Audio, Hardware
* **Topic 2 – Media:** Celebrities, Journalists, Publication
* **Topic 3 – Government:** Government, Regulations, Policy

---

## When to Use Topic Modeling

### Benefits

* Faster
* Less manual work

### Grouping Similar Content

One really great way of using topic modeling is grouping things together, such as news articles or research papers, under different topic headings.

If you've been on a news website, you'll often see that articles are grouped together under topics based on what those articles are about. These topics can either be created manually by editors and curators, or we could implement topic modeling to identify themes in the text and group them together under those topics. This speeds up the process and requires much less manual work.

### Customer Feedback and Reviews

Another example is when you have customer feedback or review data coming in. Most companies just don't have time to manually go through all of these reviews and pieces of feedback to see what their consumers are saying.

This is where topic modeling can save a significant amount of time and energy by going through all of these reviews and pulling out the key themes that customers are talking about.

### Social Listening

Topic modeling can also be used for social listening, where we monitor social media data around our brand, product, or company and quickly pull out the key topics that people are talking about in relation to our brand.
