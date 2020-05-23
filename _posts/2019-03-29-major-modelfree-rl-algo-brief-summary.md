---
layout: post
title: A brief summary of major model-free reinforcement learning algorithms
date: 2019-03-29 01:09:00
tags: reinforcement learning
---

This post I try to line the major reinforcement learning algorithms together to show how the methods for reinforcement learning have evolved. The years of first proposal for the algorithms are listed. The following figure shows how the algorithms are related with each other.

---

![Reinforcement Learning Algorithms]({{'/images/rl_algos.jpg'|relative_url}})

---
### Traditional reinforcement learning methods
* [Monte-Carlo Method](http://incompleteideas.net/book/bookdraft2018jan1.pdf)[1945]
* [SARSA (on-policy TD control)](http://incompleteideas.net/book/first/ebook/node64.html) [1988]
* [Q-learning (SARSAMAX)](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf) [1992]
* [REINFORCE](https://link.springer.com/content/pdf/10.1007%2FBF00992696.pdf) [1992]

---
### Value-based deep reinforcement learning methods

* [DQN](https://deepmind.com/research/dqn/) [2015]
* [DDQN](https://arxiv.org/abs/1509.06461) [2015]
* [Dueling DQN](https://arxiv.org/abs/1511.06581) [2015]
* [Recurrent DQN](https://arxiv.org/abs/1507.06527) [2015]
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) [2016]
* [A3C](https://arxiv.org/abs/1602.01783) [2016]
* [Distibutional DQN](https://arxiv.org/abs/1707.06887) [2017]
* [Noisy Net Exploration](https://arxiv.org/abs/1706.10295) [2018]
* [Rainbow](https://arxiv.org/abs/1710.02298) [2018]

---
### Policy-based deep reinforcement learning methods

* [TRPO](https://arxiv.org/abs/1502.05477) [2015]
* [Actor-Critic](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf) [2002]
* [DPG](http://proceedings.mlr.press/v32/silver14.pdf) [2014]
* [DDPG](https://arxiv.org/abs/1509.02971) [2015]
* [PPO](https://arxiv.org/abs/1707.06347) [2017]
* [D4PG](https://arxiv.org/pdf/1804.08617.pdf) [2018]
* [TD3](https://arxiv.org/pdf/1802.09477.pdf) [2018]
* [MADDPG](https://arxiv.org/abs/1706.02275) [2017]
