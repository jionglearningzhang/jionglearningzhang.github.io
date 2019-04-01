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

Reinforcement learning is not a new idea. It was invented back in the year of 1962. Though it blows up with suprisingly good super human level performances on varous tasks in recent years with the improvements in deep learning domain, the major ideas for solving reinforcement learning problems were established years ago.

### Monte-Carlo method  
The key idea of value-based methods is just to evalute state-action value functions from experences and then improve policy based on Q. The natual way of estimating the expected returns for a given pair of state and action is just averaging the accumulative episodic returns among episodes. In the Monte-Carlo method, the Q value gets keeping updated with the return of newly sampled experience $G_t$ with a learning rate $\alpha$, which represents how much of the memory to keep/lose. 

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (G_t - Q(S_t, A_t))$$

### TD-control (SARSA) 
While Monte-Carlo method could give unbiased estimations of Qs, it could only be applied to episodic cases because of the requirement of having $G_t$ to update Q. In fact, during the policy evaluation, while just looking as one transition $(S, A, R, S', A')$, we have already had  information to learn from and a way to estimate the return in a bootstrap fashion with this one-step reward and estimation of Q from previous iteration:  
$$ G_t \leftarrow R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) $$  

As it was done in the Monte-Carlo method, the Q function gets updated with the new estimation from the experience it learns from.
$$Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))$$  
During the policy improvement step, $\epsilon-greedy$ is used. Thus we have the full cycyle of  the generalized policy iteration for solving the problem. This is a super important method in reinforcement learning as it was said to be undoubtedly "one idea as central and novel to reinforcement learning".

### Q-learning (SARSAMAX) 
Basicly all value-based methods are based on the idea of TD-contral trying to developing various ideas to learn Q better. To be more specificly, having different ideas for estimating $G_t$ which will be used for updating Q. In TD-control method, the on-policy evaluation is used. Q-learning or SARSAMAX estimates $G_t$ with off-policy way to maximize the next step return: 
$$G_t \leftarrow R_{t+1} + \gamma \underset{a'}{\max}Q(S_{t+1}, a')$$  

$$Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} + \gamma \underset{a'}{\max}Q(S_{t+1}, a') - Q(S_t,A_t))$$  


## Kick-start with DQN
On Februry 26th 2015, DeepMind published a paper in Nature to describe the method of DQN, which revolutionized the domain of deep learning and started the era of deep reinforcement learning. The DQN employed deep nural networks for approximate functions with reinforcement learning for the first time and reached superhuman level performance on Atari 2600 games.  

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


<img src="/images/rainbow.png" width="70%" style="margin-left: auto;
  margin-right: auto;">

## References
1. [Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
2. [
Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
3. [
Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
4. [
Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
5. [
Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
6. [An Analysis of Temporal-Difference Learning
with Function Approximation](http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf)
7. [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
8. [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
9. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)