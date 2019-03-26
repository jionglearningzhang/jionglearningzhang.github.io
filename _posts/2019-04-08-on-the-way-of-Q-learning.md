---
layout: post
title: The way to learn Q better
date: 2019-04-08 01:09:00
tags: reinforcement learning
---

A big branch of solving reinforcement learning is based on learning the Q (state-action value function).  

![Reinforcement Learning Algorithms]({{'/images/deepQ.jpg'|relative_url}})

Bellman equation:


Monte-Carlo method:  
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (G_t - Q(S_t, A_t))$$

TD-control (SARSA):  
$$Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))$$  

Q-learning (SARSAMAX):  
$$Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} + \gamma max_{a'}Q(S_{t+1}, a') - Q(S_t,A_t))$$  


DQN:
Freeze the target network.
$$Q(S_t, A_t;\theta) \leftarrow Q(S_t,A_t;\theta) + \alpha (R_{t+1} + \gamma max_{a'}Q(S_{t+1}, a';\theta^-) - Q(S_t,A_t;\theta))$$  
Experience replay to decoupling correlation.  

Goal:
Better estimation of Q.

Issues of DQN:  
1. Data efficency issue because of uniform random exprience sampling  
2. Overestimation caused by max operation  
3. Exploration  

Double DQN (DDQN):  
The idea of DDQN was proposed to solve the overestimation issue in Q learning. It Decouples the evaluation and action selection in the target. The online network is used to select the action.
$$Q(S_t, A_t;\theta) \leftarrow Q(S_t,A_t;\theta) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, argmax_{a'}Q(S_t, a'; \theta});\theta^-) - Q(S_t,A_t;\theta))$$  

Dueling Q Network:  
There are cases for some certain states, choose which actions do not influence much following rewards and transitions. The dueling Q network represent this by having two streams representing the state value function $$V$$ and the advantage function $$A$$ for each action.
$$Q = Q + (A - \bar{A})$$

Prioritized Experience Replay:
Sampling with TD-error.

A3C:  
Multi-step estimation.

Distributional DQN:  
Estimate distribution of Q instead of expectation.

Noisy Net:
Replacing linear layer with stochastic layer.
$$y = b+Wx \leftarrow y = (b+Wx) + (b*\epsilon^b + W*\epsilon^Wx)$$

Combine all these complementary improvements based on DQN, the algorithm Rainbow is established.



