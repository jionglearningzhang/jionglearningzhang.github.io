---
layout: post
title: Overview of supervised meta-learning
date: 2019-08-15 01:09:00
tags: meta learning
---


> When facing a machine learning task, not every time there's enough data. Meta-learning enables the model learn through a set of tasks to learn faster when facing a new task. In this post, we will go over three major ways of the meta-learning techniques: black-box, MAML and protoNet.



Aloutgh meta-learning can also be used in the reinforcement learning domain, this post will focus on the supervised learning domain. Meta-learning is mostly used for few-shot learning. During the training time, the model gets to learn the meta-parameters from a set of tasks, and in the inference time, the model adapts fast to new tasks based on the meta-parameters with learned task-specific parameters.

The general procedure of a meta-learning algorithm works like following:

1. Have data prepared in batches of tasks.
2. For each batch of tasks, update the meta-parameters based on test data samples' loss of each task.
3. At inference / test time, learn the task-specific parameters with the training samples of the new task.
4. Predict the test samples of the new task.

The three categories of meta-learning techniques can be difreciated by how do they implement the meta-parameters and the task-specific adaptation.

Black-box: use memory enhanced models to learn model parameters as the meta-parameters and use hidden states for task-specific parameters.

(SNAIL)

MAML: this gradient-based algorithm update the model parameters as the meta-parameters during meta-training and have them transfer learning for task-specifc parameters when testing.

(MAMAL, how to train MAML)

ProtoNet: this method learns the mapping function to project sample features to a embedding space, and use nearest neighbors for predicting task-specific test samples.

(ProtoNet)




### A brief overview of meta-learning

* goal-conditioned reinforcement learning techniques that leverage the structure of the provided goal space to learn many tasks significantly faster
* meta-learning methods that aim to learn efficient learning algorithms that can learn new tasks quickly
* curriculum and lifelong learning, where the problem requires learning a sequence of tasks, leveraging their shared structure to enable knowledge transfer

 Supervised multi-task learning, black-box meta-learning
 Optimization-based meta-learning
 Few-shot learning via metric learning
 Hybrid meta-learning approaches
 Bayesian meta-learning
 Meta-learning for active learning, weakly-supervised learning, unsupervised learning
 Renforcement learning primer, multi-task RL, goal-conditioned RL
 Auxiliary objectives, state representation learning
 Hierarchical RL, curriculum generation
 Meta-RL, learning to explore
 Meta-RL and emergent phenomenon
 Model-based RL for multi-task learning, meta model-based RL
 Lifelong learning: problem statement, forward & backward transfer	
 Miscellaneous multi-task/meta-RL topics	
 Frontiers: Memorization, unsupervised meta-learning, open problems

