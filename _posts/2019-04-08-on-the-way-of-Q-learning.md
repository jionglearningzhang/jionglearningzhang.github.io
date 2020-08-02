---
layout: post
title: The way of reinforcement learning to learn Q better: from monte-carlo method to Rainbow
date: 2019-04-08 01:09:00
tags: reinforcement learning; Q learning; TD-learning
---

>A big branch of solving reinforcement learning is based on learning the Q (state-action value function). Let's take a look on how the methods evolved to learn Q better.  

[Before Deep Q Network](#befored-deep-q-network)  
[Kick-start with DQN](#dqn)  
[Enhancements of DQN](#enhance-dqn)  
[Rainbow](#rainbow)  


![Reinforcement Learning Algorithms]({{'/images/deepQ.jpg'|relative_url}})

The goal of reinforcement learning is to establish a policy, based on which to select actions at given states so that the long term return is maximized. One way is to learn the optimal state-action value function $Q^\*(S, A)$, and thus the optimal policy naturaly becomes $\pi^\*(A|S) = \underset{a'}{\arg\max}Q(S_t, a')$. With the model-free problem setting, the problem reduces as how to estimate the optimal state-action value function $Q^\*(S, A)$. Through the generalized policy iterations, policy evaluation estimates the value function, and in the policy improvement part, policy get improved with updated value function. Thus, the leftover issue is how to estimate the state-action value function $Q(S,A)$ from the experiences (state transitions) the agent gets through interact with the environment. This post talk through value-based milestone methods to get the idea of how we learn Q better.

<span style="color:red">
Value Iteration
Value iteration computes the optimal state value function by iteratively improving the estimate of V(s). The algorithm initialize V(s) to arbitrary random values. It repeatedly updates the Q(s, a) and V(s) values until they converges. Value iteration is guaranteed to converge to the optimal values.
</span>

## Before Deep Q Network

Reinforcement learning is not a new idea. It was invented back in the year of 1962. Though it blows up with surprisingly good super human level performances on various tasks in recent years with the improvements in deep learning domain, the major ideas for solving reinforcement learning problems were established years ago.

### Monte-Carlo method  
The key idea of value-based methods is just to evaluate state-action value functions from experiences and then improve policy based on Q. The natural way of estimating the expected returns for a given pair of state and action is just averaging the accumulative episodic returns among episodes. In the Monte-Carlo method, the Q value gets keeping updated with the return of newly sampled experience $G_t$ with a learning rate $\alpha$, which represents how much of the memory to keep/lose. 

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (G_t - Q(S_t, A_t))$$

### TD-control (SARSA) 
While Monte-Carlo method could give unbiased estimations of Qs, it could only be applied to episodic cases because of the requirement of having $G_t$ to update Q. In fact, during the policy evaluation, while just looking as one transition $(S, A, R, S', A')$, we have already had  information to learn from and a way to estimate the return in a bootstrap fashion with this one-step reward and estimation of Q from previous iteration:  

$$ G_t \leftarrow R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) $$  

As it was done in the Monte-Carlo method, the Q function gets updated with the new estimation from the experience it learns from.

$$Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))$$  

During the policy improvement step, $\epsilon-greedy$ is used. Thus we have the full cycyle of  the generalized policy iteration for solving the problem. This is a super important method in reinforcement learning as it was said to be undoubtedly "one idea as central and novel to reinforcement learning".

### Q-learning (SARSAMAX) 
Basically all value-based methods are based on the idea of TD-control trying to developing various ideas to learn Q better. To be more specifically, having different ideas for estimating $G_t$ which will be used for updating Q. In TD-control method, the on-policy evaluation is used. Q-learning or SARSAMAX estimates $G_t$ with off-policy way to maximize the next step return: 

$$G_t \leftarrow R_{t+1} + \gamma \underset{a'}{\max}Q(S_{t+1}, a')$$

$$Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha (R_{t+1} + \gamma \underset{a'}{\max}Q(S_{t+1}, a') - Q(S_t,A_t))$$  


## Kick-start with DQN

On February 26th 2015, DeepMind published a paper in Nature to describe the method of DQN, which revolutionized the domain of deep learning and started the era of deep reinforcement learning. The DQN employed deep neural networks for approximate functions with reinforcement learning for the first time and reached superhuman level performance on Atari 2600 games.  


TODO: problem with large nonlinear approximate function for Q  

Besides using the neural network as the approximate function, two key ideas were proposed and implemented to make the algorithm successful: target network frozen and experience replay.  

In Q-learning, the evaluation network of target and learning is the same network which gets updated every step. This setup causes the issue. DQN decouples the target network for estimating $Q(t+1, a')$ and the online learning network for estimating $Q(t, a)$ through freezing the parameters in the target network, written as $Q(t,a;\theta^-)$,  for several iterations and updated periodically by copying the parameters of the online learning network $Q(t,a;\theta)$. 

$$Q(S_t, A_t;\theta) \leftarrow Q(S_t,A_t;\theta) + \alpha (R_{t+1} + \gamma \underset{a'}{\max}Q(S_{t+1}, a';\theta^-) - Q(S_t,A_t;\theta))$$ 

The original Q-learning is an online-learning algorithm, meaning that it learn from each incoming example and then forget about it. While all the examples could be stored and replay for learning to improve the data efficiency. This idea is implemented in DQN, namely the Experience replay.  This algorithm stores the transitions in a buffer, and uniformly random sample a batch of transitions to learn from at each step. This part becomes a supervised learning problem: given a batch of samples, the network parameters are updated through minimizing the loss function which measures the difference between the on line network estimated value and the target value.

Great! Now we have a algorithm outperforms human on Atari 2600 games! This is just a start. Before we go further to go through enhancements on DQN, let's take a look at what issues of DQN still exists and how possibly further improvements can be developed.

**The general goal**: have a better estimation of Q.

**Existing Issues of DQN**:  
1. Data efficiency issue because of using uniform random experience sampling  
2. Overestimation caused by max operation  
3. Low efficiency exploration  

## Enhance DQN
With the efforts of addressing the existing issues of DQN and achieving the general goal of estimate Q better, a series of papers have been published to improve DQN. We now talk about the key ideas in these papers and what issues they addressed to prepare for putting all these improvements together to build [Rainbow](#rainbow) . 

### Double DQN (DDQN)
The idea of DDQN was proposed to solve the overestimation issue in Q learning. It Decouples the evaluation and action selection in the target. The online network is used to select the action.

$$Q(S_t, A_t;\theta) \leftarrow Q(S_t,A_t;\theta) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, \underset{a'}{\arg\max}Q(S_t, a'; \theta);\theta^-) - Q(S_t,A_t;\theta))$$  

### Dueling Q Network  
There are cases for some certain states, choose which actions do not influence much following rewards and transitions. The dueling Q network represent this by having two streams representing the state value function $$V$$ and the advantage function $$A$$ for each action.

$$Q = Q + (A - \bar{A})$$

### Prioritized Experience Replay
1. Sampling with TD-error.  
2. Intermediate factor between prioritized sampling and uniform sampling 
3. Correct estimation bias with importance sampling

### A3C  
Multi-step estimation.

<span style="color:red">  Second, we make the observation that multiple actors-learners running in parallel are likely to be exploring different parts of the environment. Moreover, one can explicitly use different exploration policies in each actor-learner
to maximize this diversity. By running different exploration policies in different threads, the overall changes being made to the parameters by multiple actor-learners applying online updates in parallel are likely to be less correlated in time than a single agent applying online updates.
Hence, we do not use a replay memory and rely on parallel
actors employing different exploration policies to perform
the stabilizing role undertaken by experience replay in the
DQN training algorithm. In addition to stabilizing learning, using multiple parallel
actor-learners has multiple practical benefits. First, we obtain a reduction in training time that is roughly linear in
the number of parallel actor-learners. Second, since we no
longer rely on experience replay for stabilizing learning we
are able to use on-policy reinforcement learning methods
such as Sarsa and actor-critic to train neural networks in a
stable way. We now describe our variants of one-step Qlearning, one-step Sarsa, n-step Q-learning and advantage
actor-critic.
Asynchronous one-step Q-learning: Pseudocode for our
variant of Q-learning, which we call Asynchronous onestep Q-learning, is shown in Algorithm 1. Each thread interacts with its own copy of the environment and at each
step computes a gradient of the Q-learning loss. We use
a shared and slowly changing target network in computing the Q-learning loss, as was proposed in the DQN training method. We also accumulate gradients over multiple
timesteps before they are applied, which is similar to using minibatches. This reduces the chances of multiple actor learners overwriting each other’s updates. 
Accumulating updates over several steps also provides some ability to
trade off computational efficiency for data efficiency.
Finally, we found that giving each thread a different exploration policy helps improve robustness. Adding diversity
to exploration in this manner also generally improves performance through better exploration. While there are many
possible ways of making the exploration policies differ we
experiment with using -greedy exploration with  periodically sampled from some distribution by each thread.
Asynchronous one-step Sarsa: The asynchronous onestep Sarsa algorithm is the same as asynchronous one-step Q-learning as given in Algorithm 1 except that it uses a different target value for Q(s, a). 
The target value used by one-step Sarsa is r + γQ(s 0 , a0 ; θ −) where a 0 is the action taken in state s 0 (Rummery & Niranjan, 1994; Sutton & Barto, 1998). 
We again use a target network and updates accumulated over multiple timesteps to stabilize learning.
Asynchronous n-step Q-learning: Pseudocode for our
variant of **multi-step Q-learning** is shown in Supplementary
Algorithm S2. The algorithm is somewhat unusual because
it operates in the forward view by explicitly computing nstep returns, as opposed to the more common backward
view used by techniques like eligibility traces (Sutton &
Barto, 1998). We found that using the forward view is easier when training neural networks with momentum-based
methods and backpropagation through time. In order to
compute a single update, the algorithm first selects actions
using its exploration policy for up to tmax steps or until a
terminal state is reached. This process results in the agent
receiving up to tmax rewards from the environment since
its last update. The algorithm then computes gradients for
n-step Q-learning updates for each of the state-action pairs
encountered since the last update. Each n-step update uses
the longest possible n-step return resulting in a one-step
update for the last state, a two-step update for the second
last state, and so on for a total of up to tmax updates. The
accumulated updates are applied in a single gradient step.</span>

### Distributional DQN  
Estimate distribution of Q instead of expectation.

<span style="color:red"> In this paper we argue for the fundamental importance of the value distribution: the distribution of the random return received by a reinforcement learning agent. This is in contrast to the common approach to reinforcement learning which models the expectation of this return, or value.</span>

<span style="color:red">In this section we propose an algorithm based on the distributional Bellman optimality operator. In particular, this will require choosing an approximating distribution. Although the Gaussian case has previously been considered (Morimura et al., 2010a; Tamar et al., 2016), to the best of our knowledge we are the first to use a rich class of parametric distributions. 4.1. Parametric Distribution We will model the value distribution using a discrete distribution parametrized by N ∈ N and VMIN, VMAX ∈ R, and whose support is the set of atoms {zi = VMIN + i4z : 0 ≤ i < N}, 4z := VMAX−VMIN N−1 . In a sense, these atoms are the “canonical returns” of our distribution. The atom probabilities are given by a parametric model θ : X × A → R N Zθ(x, a) = zi w.p. pi(x, a) := e θi(x,a) P j e θj (x,a) . The discrete distribution has the advantages of being highly expressive and computationally friendly (see e.g. Van den Oord et al., 2016). 4.2. Projected Bellman Update Using a discrete distribution poses a problem: the Bellman update T Zθ and our parametrization Zθ almost always have disjoint supports. From the analysis of Section 3 it would seem natural to minimize the Wasserstein metric (viewed as a loss) between T Zθ and Zθ, which is also conveniently robust to discrepancies in support. However, a second issue prevents this: in practice we are typically restricted to learning from sample transitions, which is not possible under the Wasserstein loss (see Prop. 5 and toy results in the appendix).</span>


### Noisy Net
 <span style="color:blue">We introduce NoisyNet, a deep reinforcement learning agent with parametric noise
added to its weights, and show that the induced stochasticity of the agent’s policy
can be used to aid efficient exploration. The parameters of the noise are learned
with gradient descent along with the remaining network weights. NoisyNet is
straightforward to implement and adds little computational overhead. We find that
replacing the conventional exploration heuristics for A3C, DQN and Dueling agents
(entropy reward and -greedy respectively) with NoisyNet yields substantially
higher scores for a wide range of Atari games, in some cases advancing the agent
from sub to super-human performance.</span>

 <span style="color:blue">We propose a simple alternative approach, called NoisyNet, where learned perturbations of the
network weights are used to drive exploration. The key insight is that a single change to the weight
vector can induce a consistent, and potentially very complex, state-dependent change in policy over
multiple time steps – unlike dithering approaches where decorrelated (and, in the case of -greedy,
state-independent) noise is added to the policy at every step. The perturbations are sampled from
a noise distribution. The variance of the perturbation is a parameter that can be considered as
the energy of the injected noise. These variance parameters are learned using gradients from the
reinforcement learning loss function, along side the other parameters of the agent. The approach
differs from parameter compression schemes such as variational inference (Hinton & Van Camp,
1993; Bishop, 1995; Graves, 2011; Blundell et al., 2015; Gal & Ghahramani, 2016) and flat minima
search (Hochreiter & Schmidhuber, 1997) since we do not maintain an explicit distribution over
weights during training but simply inject noise in the parameters and tune its intensity automatically.
Consequently, it also differs from Thompson sampling (Thompson, 1933; Lipton et al., 2016) as the
distribution on the parameters of our agents does not necessarily converge to an approximation of a
posterior distribution.</span>

 <span style="color:blue">At a high level our algorithm is a randomised value function, where the functional form is a neural
network. Randomised value functions provide a provably efficient means of exploration (Osband
et al., 2014). Previous attempts to extend this approach to deep neural networks required many
duplicates of sections of the network (Osband et al., 2016). By contrast in our NoisyNet approach
while the number of parameters in the linear layers of the network is doubled, as the weights are a
simple affine transform of the noise, the computational complexity is typically still dominated by
the weight by activation multiplications, rather than the cost of generating the weights. Additionally,
it also applies to policy gradient methods such as A3C out of the box (Mnih et al., 2016). Most
recently (and independently of our work) Plappert et al. (2017) presented a similar technique where
constant Gaussian noise is added to the parameters of the network. Our method thus differs by the
ability of the network to adapt the noise injection with time and it is not restricted to Gaussian noise
distributions. We need to emphasise that the idea of injecting noise to improve the optimisation
process has been thoroughly studied in the literature of supervised learning and optimisation under
different names (e.g., Neural diffusion process (Mobahi, 2016) and graduated optimisation (Hazan
et al., 2016)). These methods often rely on a noise of vanishing size that is non-trainable, as opposed
to NoisyNet which tunes the amount of noise by gradient descent.</span>

 <span style="color:blue">NoisyNets are neural networks whose weights and biases are perturbed by a parametric function
of the noise. These parameters are adapted with gradient descent. More precisely, let y = fθ(x)
be a neural network parameterised by the vector of noisy parameters θ which takes the input x and
outputs y. We represent the noisy parameters θ as θ
def = µ + Σ  ε, where ζ
def = (µ, Σ) is a set of
vectors of learnable parameters, ε is a vector of zero-mean noise with fixed statistics and  represents
element-wise multiplication. The usual loss of the neural network is wrapped by expectation over the
noise ε: L¯(ζ)
def = E [L(θ)]. Optimisation now occurs with respect to the set of parameters ζ.
Consider a linear layer of a neural network with p inputs and q outputs, represented by
y = wx + b, (8)
where x ∈ R
p
is the layer input, w ∈ R
q×p
the weight matrix, and b ∈ R
q
the bias. The corresponding
noisy linear layer is defined as:
y
def = (µ
w + σ
w  ε
w)x + µ
b + σ
b  ε
b
, (9)
where µ
w + σ
w  ε
w and µ
b + σ
b  ε
b
replace w and b in Eq. (8), respectively. The parameters
µ
w ∈ R
q×p
, µ
b ∈ R
q
, σ
w ∈ R
q×p
and σ
b ∈ R
q
are learnable whereas ε
w ∈ R
q×p
and ε
b ∈ R
q
are
noise random variables (the specific choices of this distribution are described below). We provide a
graphical representation of a noisy linear layer in Fig. 4 (see Appendix B).
We now turn to explicit instances of the noise distributions for linear layers in a noisy network.
We explore two options: Independent Gaussian noise, which uses an independent Gaussian noise
entry per weight and Factorised Gaussian noise, which uses an independent noise per each output
and another independent noise per each input. The main reason to use factorised Gaussian noise is
to reduce the compute time of random number generation in our algorithms. This computational
overhead is especially prohibitive in the case of single-thread agents such as DQN and Duelling. For
this reason we use factorised noise for DQN and Duelling and independent noise for the distributed
A3C, for which the compute time is not a major concern.</span>

 <span style="color:blue">(a) Independent Gaussian noise: the noise applied to each weight and bias is independent, where
each entry ε
w
i,j (respectively each entry ε
b
j
) of the random matrix ε
w (respectively of the random
vector ε
b
) is drawn from a unit Gaussian distribution. This means that for each noisy linear layer,
there are pq + q noise variables (for p inputs to the layer and q outputs).

 <span style="color:blue">(b) Factorised Gaussian noise: by factorising ε
w
i,j , we can use p unit Gaussian variables εi for noise
of the inputs and and q unit Gaussian variables εj for noise of the outputs (thus p + q unit
Gaussian variables in total). Each ε
w
i,j and ε
b
j
can then be written as:
ε
w
i,j = f(εi)f(εj ), (10)
ε
b
j = f(εj ), (11)
where f is a real-valued function. In our experiments we used f(x) = sgn(x)
p
|x|. Note that
for the bias Eq. (11) we could have set f(x) = x, but we decided to keep the same output noise
for weights and biases.</span>

 <span style="color:blue">Since the loss of a noisy network, L¯(ζ) = E [L(θ)], is an expectation over the noise, the gradients are
straightforward to obtain:
∇L¯(ζ) = ∇E [L(θ)] = E [∇µ,ΣL(µ + Σ  ε)] . (12)
We use a Monte Carlo approximation to the above gradients, taking a single sample ξ at each step of
optimisation:
∇L¯(ζ) ≈ ∇µ,ΣL(µ + Σ  ξ). (13)</span>

 <span style="color:blue">We now turn to our application of noisy networks to exploration in deep reinforcement learning. Noise
drives exploration in many methods for reinforcement learning, providing a source of stochasticity
external to the agent and the RL task at hand. Either the scale of this noise is manually tuned across a
wide range of tasks (as is the practice in general purpose agents such as DQN or A3C) or it can be
manually scaled per task. Here we propose automatically tuning the level of noise added to an agent
for exploration, using the noisy networks training to drive down (or up) the level of noise injected
into the parameters of a neural network, as needed.
A noisy network agent samples a new set of parameters after every step of optimisation. Between
optimisation steps, the agent acts according to a fixed set of parameters (weights and biases). This
ensures that the agent always acts according to parameters that are drawn from the current noise
distribution.</span>

Replacing linear layer with stochastic layer.

$$y = b+Wx \leftarrow y = (b+Wx) + (b*\epsilon^b + W*\epsilon^Wx)$$

Combine all these complementary improvements based on DQN, the algorithm Rainbow is established.

## Rainbow
All of the improvements mentioned previously are independent and complementary to each other. In this paper, all of these developments are combined to formulate the strongest agent: Rainbow. it is composed as following:  
1. Use distibutional multi-step Q-distribution KL-divergence as the loss function.    
2. Q functions are estimated with dueling network, whose outputs are categorical distributions.  
3. Double DQN was used for policy evaluation.  
4. Priorized experience replay with sampling using KL loss as the proxy (instead of the TD-error in the non-distributional case).   
5. As used in Noisy net, linear layers in Q networks are replaced by stochastic layers.

 <span style="color:blue">
 Ablation studies. Since Rainbow integrates several different ideas into a single agent, we conducted additional experiments to understand the contribution of the various components, in the context of this specific combination.
To gain a better understanding of the contribution of each
component to the Rainbow agent, we performed ablation
studies. In each ablation, we removed one component from
the full Rainbow combination. Figure 3 shows a comparison for median normalized score of the full Rainbow to six
ablated variants. Figure 2 (bottom row) shows a more detailed breakdown of how these ablations perform relative to
different thresholds of human normalized performance, and
Figure 4 shows the gain or loss from each ablation for every
game, averaged over the full learning run.
Prioritized replay and multi-step learning were the two
most crucial components of Rainbow, in that removing either component caused a large drop in median performance.
Unsurprisingly, the removal of either of these hurt early performance. Perhaps more surprisingly, the removal of multistep learning also hurt final performance. Zooming in on individual games (Figure 4), we see both components helped
almost uniformly across games (the full Rainbow performed
better than either ablation in 53 games out of 57).
Distributional Q-learning ranked immediately below the
previous techniques for relevance to the agent’s performance. Notably, in early learning no difference is apparent, as shown in Figure 3, where for the first 40 million
frames the distributional-ablation performed as well as the
full agent. However, without distributions, the performance
of the agent then started lagging behind. When the results are
separated relatively to human performance in Figure 2, we
see that the distributional-ablation primarily seems to lags
on games that are above human level or near it.
In terms of median performance, the agent performed
better when Noisy Nets were included; when these are removed and exploration is delegated to the traditional -
greedy mechanism, performance was worse in aggregate
(red line in Figure 3). While the removal of Noisy Nets produced a large drop in performance for several games, it also
provided small increases in other games (Figure 4).
In aggregate, we did not observe a significant difference
when removing the dueling network from the full Rainbow.
The median score, however, hides the fact that the impact
of Dueling differed between games, as shown by Figure 4.
Figure 2 shows that Dueling perhaps provided some improvement on games with above-human performance levels
(# games > 200%), and some degradation on games with
sub-human performance (# games > 20%).
Also in the case of double Q-learning, the observed difference in median performance (Figure 3) is limited, with the
component sometimes harming or helping depending on the
game (Figure 4). To further investigate the role of double Qlearning, we compared the predictions of our trained agents
to the actual discounted returns computed from clipped rewards. Comparing Rainbow to the agent where double Qlearning was ablated, we observed that the actual returns are
often higher than 10 and therefore fall outside the support
of the distribution, spanning from −10 to +10. This leads to
underestimated returns, rather than overestimations. We hypothesize that clipping the values to this constrained range
counteracts the overestimation bias of Q-learning. Note,
however, that the importance of double Q-learning may increase if the support of the distributions is expanded.
In the appendix, for each game we show final performance
and learning curves for Rainbow, its ablations, and baselines.
Discussion
We have demonstrated that several improvements to DQN
can be successfully integrated into a single learning algorithm that achieves state-of-the-art performance. Moreover,
we have shown that within the integrated algorithm, all but
one of the components provided clear performance benefits. There are many more algorithmic components that we
were not able to include, which would be promising candidates for further experiments on integrated agents. Among
the many possible candidates, we discuss several below.
We have focused here on value-based methods in the
Q-learning family. We have not considered purely policybased RL algorithms such as trust-region policy optimisa-
</span>

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