---
layout: post
title: A brief overview of deep learning in ranking problems
date: 2020-01-31 01:09:00
tags: search ranking
---


> Learning to rank is playing an important role in
search, recommendation, and
many other applications. 
We will walk through the learning-to-rank (LTR) problems and major algorithms in this post, and discuss about how deep learning helps to achieve better performance.

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

While the independently predicted score does not mean much in a ranking problem, since the problem cares more about the relative position of each candidate.
Pair-wise strategy groups ranking candidates into pairs and then try to predict which one has a higher relevance.
The major question then is how to define a list-wise loss function, representing the difference between the ranking list output by
a ranking model and the ranking list given as ground truth.

LambdaRank
LambdaMart

Instead of take one or two candidates as an instance, List-wise takes the whole list as an instance and tries to predict the ranking order directly.
* ListNet
* softmax_loss, 
* approxNDCG


# How deep learning improves ranking further

## ListNet

Based on the top one probability, with a Neural Network as a model and Gradient Descendant as optimization, the authors are able to optimize the listwide loss function.

In ListNet, the ranking function, based on the Neural Network model w is denoted as fw. Given a feature vector xj(i), fw(xj(i))will assign an score for it.

If we take the exponential function as , we can rewrite the top one probability of document dj(i)as:

![Image for post](https://miro.medium.com/max/60/1*qFNDaKTdTp5GVdhle6a6GQ.png?q=20)

![Image for post](https://miro.medium.com/max/358/1*qFNDaKTdTp5GVdhle6a6GQ.png)

top one probability of document dj

The learning algorithm for ListNet can be described as:

> **Input**: training data (pairs of document, features and target scores)
>
> **Parameter**: number of iterations T and learning rate η
>
> Initialise parameter ω (weights)
>
> **for** t = 1 **to** T **do**
>
> **for** i = 1 **to** m **do**
>
> Input x(i) of query q(i) to Neural Network and compute score list z(i)(fω) with current ω
>
> Compute gradient 𝛥ω
>
> Update ω = ω − η × 𝛥ω
>
> **end for**
>
> **end for**
>
> Output Neural Network model ω



deep learning features