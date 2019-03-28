---
layout: post
title: The way to learn Q better
date: 2019-04-08 01:09:00
tags: reinforcement learning
---

>A big branch of solving reinforcement learning is based on learning the Q (state-action value function). Let's take a look on how the methods evolved to learn Q better.  

[Before Deep Q Network](#befored-deep-q-network)  
[Kick-start with DQN](#dqn)  
[Enhancements of DQN](#enhance-dqn)  
[Rainbow](#rainbow)  


![Reinforcement Learning Algorithms]({{'/images/deepQ.jpg'|relative_url}})

The goal of reinforcement learning is to establish a policy, based on which to select actions at given states so that the long term return is maximized. One way is to learn the optimal state-action value function $Q^\*(S, A)$, and thus the optimal policy naturaly becomes $\pi^\*(A|S) = \underset{a'}{\arg\max}Q(S_t, a')$. With the model-free problem setting, the problem reduces as how to estimate the optimal state-action value function $Q^\*(S, A)$. Through the generalized policy iterations, policy evaluation estimates the value function, and in the policy improvement part, policy get improved with updated value function. Thus, the leftover issue is how to estimate the state-action value function $Q(S,A)$ from the experiences (state transitions) the agent gets through interact with the environment. This post talk through value-based milestone methods to get the idea of how we learn Q better.
[//]: # (TODO: add Q convergence to Q* )


## Before Deep Q Network

### Monte-Carlo method  

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (G_t - Q(S_t, A_t))$$

### TD-control (SARSA) 

$$ G_t \leftarrow R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) $$  

$$Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))$$  

### Q-learning (SARSAMAX) 

$$G_t \leftarrow R_{t+1} + \gamma \underset{a'}{\max}Q(S_{t+1}, a')$$  

$$Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} + \gamma \underset{a'}{\max}Q(S_{t+1}, a') - Q(S_t,A_t))$$  


## Kick-start with DQN
Freeze the target network.

$$Q(S_t, A_t;\theta) \leftarrow Q(S_t,A_t;\theta) + \alpha (R_{t+1} + \gamma \underset{a'}{\max}Q(S_{t+1}, a';\theta^-) - Q(S_t,A_t;\theta))$$ 
 
Experience replay to decoupling correlation.  

Goal:
Better estimation of Q.

Issues of DQN:  
1. Data efficency issue because of using uniform random exprience sampling  
2. Overestimation caused by max operation  
3. Exploration  

## Enhance DQN
### Double DQN (DDQN)
The idea of DDQN was proposed to solve the overestimation issue in Q learning. It Decouples the evaluation and action selection in the target. The online network is used to select the action.

$$Q(S_t, A_t;\theta) \leftarrow Q(S_t,A_t;\theta) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, \underset{a'}{\arg\max}Q(S_t, a'; \theta);\theta^-) - Q(S_t,A_t;\theta))$$  

### Dueling Q Network  
There are cases for some certain states, choose which actions do not influence much following rewards and transitions. The dueling Q network represent this by having two streams representing the state value function $$V$$ and the advantage function $$A$$ for each action.

$$Q = Q + (A - \bar{A})$$

### Prioritized Experience Replay
Sampling with TD-error.  
Intermidieate factor between prioritized sampling and uniform sampling 
Correct estimation bias with importance sampling

### A3C  
Multi-step estimation.

### Distributional DQN  
Estimate distribution of Q instead of expectation.

### Noisy Net
Replacing linear layer with stochastic layer.

$$y = b+Wx \leftarrow y = (b+Wx) + (b*\epsilon^b + W*\epsilon^Wx)$$

Combine all these complementary improvements based on DQN, the algorithm Rainbow is established.

## Rainbow
All of the improvements mentioned previously are independent and complementary to each other. In the paper, all of these developments are combined to formulate the strongest agent: Rainbow. it is composed as following:  
1. Use distibutional multi-step Q-distribution KL-divergence as the loss function.    
2. Q functions are estimated with dueling network, whose outputs are categorical distributions.  
3. Double DQN was used for policy evaluation.  
4. Priorized experience replay with sampling using KL loss as the proxy (instead of the TD-error in the non-distributional case).   
5. As used in Noisy net, linear layers in Q networks are replaced by stochastic layers.


<img src="/images/rainbow.png" width="60%" style="margin-left:auto; margin-right:auto;">
