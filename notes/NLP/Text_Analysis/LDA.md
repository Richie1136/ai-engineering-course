# Latent Dirichlet Allocation (LDA)

## Overview

Suppose we have a collection of news articles that we want to classify into different topics. These topics could include politics, sports, business, and more.

How do we decide which document belongs to which topic? We can look at the words used throughout the document. For example, words such as **ball**, **player**, **match**, and **game** are commonly associated with sports. If these words appear many times in a document, we can probably guess that the article is about sports.

Some words are more relevant to a particular topic, while others can appear in multiple topics. For example, the word **"crash"** could relate to traffic news or financial news.

We say that the topic is **latent** within the document. We must examine the words within it to discover the topic of that piece of text.

---

## Example

**Document:**

> Amidst the growing concerns about climate change, politicians are facing increasing pressure to implement policies that address environmental challenges and promote sustainability.

A document can obviously contain words from more than one topic, but we make the assumption that it is mainly focused on a single topic, with only a few words relating to other topics. This is our **Dirichlet distribution**.

In this example:

* **Climate Change:** climate, change, environmental, sustainability
* **Politics:** politicians, policies

---

## How LDA Works

LDA works as an iterative process.

First, we specify the number of topics that we want, represented by **k**.

During the first iteration of the algorithm, the words in each document are randomly assigned to one of the **k** topics.

We then move to the second iteration, where we go through each individual word in the document. We assume that all of the other words are correctly assigned to a topic, and we try to correct the current word's topic assignment.

LDA corrects the assignment by looking at:

* The proportion of words in the current document assigned to a topic.
* The number of times that word is assigned to a specific topic in other documents.

The algorithm repeats this process multiple times until it reaches a steady state. It then produces the final topic assignments for us to investigate.
