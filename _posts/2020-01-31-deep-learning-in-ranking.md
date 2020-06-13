---
layout: post
title: A brief overview of deep learning in ranking problems
date: 2020-01-31 01:09:00
tags: search ranking
---


Learning to rank is playing an important role in
search, recommendation, and
many other applications. 
We will walk through the learning-to-rank (LTR) problems and major algorithems in this post, and discuss about how deep learning helps to acheive better performance.

For the rest of this post, we will take search problem as an example to talk though. 
%search ranking problem definition
For the domain of search task, Learning to rank is a
task as follows. Given a pool of documents as search result candidates, and a user input query, the ranking function assigns a score to each document, and ranks
the documents in descending order of the scores. The ranking order represents relative relevance of documents with
respect to the query (maybe the user as well). In learning, a number of queries are
provided; each query is associated with a perfect ranking
list of documents; a ranking function is then created using
the training data, such that the model can precisely predict
the ranking lists in the training data.


% solving strategy / machine learning problem definition
Given the above problem definition, we could tackle this problem with three strategies:
with Point-wise prediction, pair-wise prediction or list-wise prediction.
Point-wise prediction strategy takes each candidate and try to predict the relevance score independently.
Thus, the ranking problem is converted to a supervised learning problem, and common supervised learning algorithms
can be directly applied.

While the independently predicted score does not mean much in a ranking problem, since the problem cares more about the reletive position of each candidate.
Pair-wise strategy groups ranking candidates into pairs and then try to predict which one has a higher relevance.
The major question then is how to define a listwise loss function, representing the difference between the ranking list output by
a ranking model and the ranking list given as ground truth.

LambdaRank
LambdaMart

Instead of take one or two candidates as an instance, List-wise takes the whole list as an instance and tries to predict the ranking order directly.
* ListNet
* softmax_loss, 
* approxNDCG


# How deep learning improves ranking further

deep learning features
