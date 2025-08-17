<!-- markdownlint-disable MD001 MD010 MD024 MD025 MD041 MD049 -->

> # Notes

Modern chatbot systems do mimic human specialization, either via:

- **Mixture of Experts (MoE)**: One big LLM with specialized internal modules

- **Router + Specialized LLMs**: A system that picks the right LLM for the task

> # YSDA Lectures

# YSDA Lecture 1 - MDP, CEM

$\pi_{\theta}(a | s)$ - policy.

## DAGGER

Human-in-the-loop.

Can't become better than expert.

## Markov Decision Process (MDP)

Let:

- $ s \in \mathcal{S} $
- $ a \in \mathcal{A} $
- $ \pi(a \mid s) $ – policy
- $ p(s\_{t+1} \mid s_t, a_t) $ – dynamics (transition probability)
- $ r(s*t, a_t, s*{t+1}) $ – reward function
- $ p(s_0) $ – initial state distribution

MDP satisfies **Markov Property**.

Define a trajectory:

- $ \tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots) $

The probability of a trajectory under policy $ \pi $:

$$
p(\tau \mid \pi) = p(s_0) \prod_{t=0}^{\infty} \pi(a_t \mid s_t) \, p(s_{t+1} \mid s_t, a_t)
$$

> Agent that strives for the max total reward should do the task best.

### Operations on Reward

- Scale by positive

- **Reward shaping**: $ r'(s*t, a_t, s*{t+1}) = r(s*t, a_t, s*{t+1}) + \Phi(s\_{t+1}) - \Phi(s_t) $

## Cross-Entropy Method (CEM)

Set initial policy (mb _uniformal_ among actions).

![Cross-Entropy Method](notes_images/crossentropy_method.png)

- Sample N games with that policy

- Get M best sessions (**elites**)

  $$
  \text{Elite} = [(s_0, a_0), (s_1, a_1), \ldots, (s_k, a_k)]
  $$

- Calculate new policy based on elites

- Policy = $(1 - \alpha)$ Policy + $\alpha$ NewPolicy

### Tabular Cross-Entropy Method

Policy is a matrix:

$$
\pi(a \mid s) = A_{s, a}
$$

### Approximate Cross-Entropy Method

Can’t set $\pi(a \mid s)$ explicitly.

Policy is approximated using:

- Neural network predicts $\pi_{w}(a \mid s)$ given $s$
- Linear model / Random Forest / …

New policy:

$$
\pi = \arg\max_{\pi} \sum_{s_i, a_i \in \mathrm{Elite}} \log \pi\bigl(a_i \mid s_i\bigr)
$$

> If **action space** is continuous $\to$ regression instead of classification.

## Learning Tricks

1. Remember sessions of 3-5 previous iterations

2. Regularize with entropy (for exploration)

3. Parallelize sampling

# YSDA Lecture 2 - Dynamic Programming

## Choices of Reward Function

All of them are MDP:

1. $r_t := r(a_t, s_t)$ $\leftarrow$ we'll be using it

2. $r_t := r(s_{t+1}, a_t, s_t)$

3. $r_t \sim p(r \mid s_t, a_t, s_{t+1})$ $\leftarrow$ most general form

## Definitions

### Basic

$$
G_t = \sum_{t'=t}^{\infty} \gamma^{t' - t} r_{t'}
$$

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]
$$

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right] = \mathbb{E}_{a_t \sim \pi} \left[ Q^\pi(s_t, a_t) \right]
$$

### Recurrent Relations

$$
Q^\pi(s, a) = \mathbb{E}_{s_{t+1}} \left[ r_t + \gamma V^\pi(s_{t+1}) \right]
$$

$$
Q^\pi(s, a) = \mathbb{E}_{s_{t+1}, a_{t+1} \sim \pi} \left[ r_t + \gamma Q^\pi(s_{t+1}, a_{t+1}) \right]
$$

### Optimal Policy

For all $ \pi, s, a$:

$$  Q^{\pi^*}(s, a) \ge Q^\pi(s, a) $$

$$
\pi^*(s) = \arg\max_a Q^{\pi^*}(s, a)
$$

### Bellman Optimality Equation

$$
Q^*(s_t, a) = \mathbb{E}_{s_{t+1}} \left[ r_t + \gamma \max_{a'} Q^*(s_{t+1}, a') \right]
$$

# YSDA Lecture 3 - Tabular RL

## Model-based/free RL

> We might not know actual **environment model** $P(s', r | s, a) \to$ **Model-free** RL.

In **Model-based** transition probabilities are known $\to$ just use Value Iteration.

## Value Learning

The **intuition**: we learn `Q`s at each state so that we get knowledge of what value ($G_t$) we'll get for each action.

> All value learning happens under Bellman Equation. But what if our objectrive different and we don't want fading by $\gamma$ rewards? $\to$ **Credit Assignment**.

Initialize $Q(s, a)$ with 0.

In loop:

- Sample $<...>$ from environment

- Compute: $ \hat{Q}(s, a) $

- (_Tabular_) Update: $ Q(s, a) \gets \alpha \cdot \hat{Q}(s, a) + (1 - \alpha) Q(s, a) $

### Q Learning

> No matter what _policy strategy_ and actual action is we consider only the best path $\to$ off-policy

$$ \hat{Q}(s, a) = r(s, a) + \gamma \max_{a_i}{Q(s', a_i)}$$

#### 1. Monte-Carlo

![alt text](notes_images/q_learning.png)

#### 2. Temporal Difference

![alt text](notes_images/td_learning.png)

### SARSA

$$ \hat{Q}(s, a) = r(s, a) + \gamma Q(s', a') $$

- $a'$ is action chosen by policy

### Expected Value SARS(A)

$$ \hat{Q}(s, a) = r(s, a) + \gamma \cdot \mathbb{E}_{a' \sim \pi(a' \mid s')}{Q(s', a')} $$

## Value Learning Policy Strategies

Policy derived from values using one of the following strategies:

- **Greedy policy**:  
  Always select the best action: $ a = \arg\max_a Q(s, a) $

- **$\epsilon$-greedy policy**:  
  With probability $ \epsilon $, take a random action (exploration), with $ 1 - \epsilon $, take greedy action (exploitation)

- **Softmax / Boltzmann policy**:  
  Choose actions stochastically based on their relative Q-values:  
  $ \pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_b e^{Q(s,b)/\tau}} $,  
  where $ \tau $ is the temperature (controls exploration)

- **Greedy in the limit of infinite exploration (GLIE)**:  
  Exploration decays over time, but all actions are eventually tried infinitely often.

## On/Off-Policy

### On-Policy Learning

- Learns about: The same policy that is being used to make decisions.

- Updates on _actually_ taken by the policy actions.

Example: SARSA

### Off-Policy Learning

- Learns about: A different (target) policy than the one used to collect data.

- Updates as if it had acted _greedily_, even if it behaved differently.

> Even if the agent **explores**, it assumes it will behave _greedily in the future_.

Example: Q-learning, Expected Value SARSSA

## N-step Algorithms

They all are _on-policy_.

$\uparrow N$ $\to$ more noise and variance, but faster learning $\to$ great at beginning of learning

### N-step SARSA

$$
\hat{Q}(s_t, a_t) = \left[ \sum_{\tau = t}^{\tau < t + n} \gamma^{\tau - t} \, r(s_{\tau}, a_{\tau}) \right] + \gamma^n Q(s_{t+n}, a_{t+n})
$$

### N-step Q-learning

$$
\hat{Q}(s_t, a_t) = \left[ \sum_{\tau = t}^{\tau < t + n} \gamma^{\tau - t} \, r(s_{\tau}, a_{\tau}) \right] + \gamma^n \max_{a} Q(s_{t+n}, a)
$$

## TDLAMBDA???

-

# Lecture 4 - Deep RL

## Approximation Value Learning

Update:

$$ Q(s_t, a_t) \gets \alpha \cdot \hat{Q}(s_t, a_t) + (1 - \alpha) \cdot Q(s_t, a_t) $$

$$ Q(s, a) \gets Q(s_t, a_t) + \alpha \cdot ( \hat{Q}(s_t, a_t) - Q(s_t, a_t) ) $$

- _Assume_ 1-step TD Q-Learning

$$ Q(s, a) \gets Q(s_t, a_t) + \alpha \cdot ( r_t + \max_{a'}{Q(s_{t + 1}, a')} - Q(s_t, a_t) ) $$

This is equivalent to minimizing (MSE) the TD error:

$$L_t = (r_t + \max_{a'}{Q(s_{t + 1}, a')} - Q(s_t, a_t)) ^ 2 $$

But $r_t + \max_{a'}{Q(s_{t + 1}, a')}$ is a **target** and we don't want to calculate gradient on it $\to$ we treat target as fixed $\to$ **stop gradinet**:

$$L_t = (r_t + \max_{a'}{Q(s_{t + 1}, a'; \theta^-)} - Q(s_t, a_t; \theta)) ^ 2 $$

$$L_t = (Q^-_{target} - Q(s_t, a_t; \theta)) ^ 2 $$

- $Q^-_{target} \equiv r_t + \max_{a'}{Q(s_{t + 1}, a'; \theta^-)}$
- $r_t$ - constant
- $Q(s_{t + 1}, a'; \theta^-)$ fixed by no grad $\theta^-$

> “I think I’ll get reward $r_t$ now and then the best future I can hope for is whatever my $Q$-function currently says about the next state.”

$$
\nabla_\theta \mathcal{L} = -\left( Q^-_{target} - Q(s_t, a_t; \theta) \right) \cdot \nabla_\theta Q(s_t, a_t; \theta)
$$

> Note: tabular approximation can be also used as approximation

## DRL Architectures

**DQN** - Deep Q-networks

![alt text](notes_images/drl_arch.png)

## Training Sample i.i.d Problem

### Multi-Agent

![alt text](notes_images/multi_agent.png)

### Experience Replay Buffer

![alt text](notes_images/replay_buffer.png)

## N-gram

> **Idea**: DRL's input consists of N recent states (and actions).

## Target Network

> **Idea**: To keep $Q^-_{target}$ fixed we use network with **frozen** weights $\theta^-$ to compute the target.

### Hard target network

Update $ \theta^- $ every **n** steps and set its weights as $ \theta $

### Soft target network

Update $ \theta^- $ every step:  

$$
\theta^- = (1 - \alpha)\theta^- + \alpha \theta
$$

## Actor & Critic

- **Actor** $\mu$: chooses $a'$
  $$ L_{Actor} = (Q(s, a) - Q^-_{target}) ^ 2 $$

- **Critic** $Q$: given environment and chosen $a$ predicts $Q(s, a)$
  $$ L_{Critic} = -Q(s, \mu(s)) $$

Both of them have their target network $\to$ 4 DQN.

## Double Q-Learning

["Deep Reinforcement Learning with Double Q-learning", 2016](https://arxiv.org/abs/1509.06461)

### Overestimation Problem

If the Q-values are noisy, then $max$ tends to overestimate the true value $\to$ by **Jensen Inequality** we'll overestimate.

### Idea

$$
\max_{a'} Q(s', a')
$$

$$
Q(s', \arg\max_{a'} Q(s', a'))
$$

> **Idea**: evaluate $Q$ and select $Q$ are different

$$
Q^{(2)}(s_{t+1}, \arg\max_{a'} Q^{(1)}(s_{t+1}, a'))
$$

### Double Q-Learning

- Maintain two Q-functions: $ Q^{(1)} $, $ Q^{(2)} $

- On each update with 50% probability _select_ action using $ Q^{(j)} $, _evaluate_ with $ Q^{(i)} $:
  $$
  \text{target} = r_t + \gamma Q^{(i)}(s_{t+1}, \arg\max_{a'} Q^{(j)}(s_{t+1}, a'))
  $$

- Update only the chosen $ Q^{(j)} $ by minimizing:
   $$
   \mathcal{L} = \left( \text{target} - Q^{(j)}(s_t, a_t) \right)^2
   $$

### Double DQN (Deep Version)

- **online $ Q_\theta $ network** selects the best action
- **target $ Q_{\theta^-} $ network** evaluates that action

$$
\text{target} = r_t + \gamma Q_{\theta^-}(s_{t+1}, \arg\max_{a'} Q_\theta(s_{t+1}, a'))
$$

## Prioritized Experience Replay

["Prioritized Experience Replay", 2016](https://arxiv.org/abs/1511.05952)

Simple Experience Replay treats all transitions equally, but not all of them are _useful_.

> **Idea**: sample more frequently those transitions with higher “learning potential.”

- TD error: $ \delta_i = r + \gamma \max_{a'} Q(s', a') - Q(s, a) $

- Priority: $ p_i = |\delta_i| + \epsilon $

Transitions are sampled with probability:

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$

### Roulette Problem

Based on loss we would only work with states when green has won $\to$ bias.

### Sampling Correction

$\to$ Use **importance sampling (IS) weights** to correct for the bias:

$$
w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta
$$

$$
\mathcal{L}_i = w_i \cdot \left( Q(s, a) - \text{target} \right)^2
$$

- Potential overfitting is mitigated by annealing $\beta$ over time

## Duelling DQN

["Dueling Network Architectures for Deep Reinforcement Learning", 2016](https://arxiv.org/abs/1511.06581)

We want to know:

- how important is current state itself $\to$ State-value function $V(s)$

- how important each action in current state $\to$ **Advantage function** $A(s, a) = Q(s, a) - V(s)$

To avoid the _identifiability issue_, we normalize the advantage function:

$$
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)
$$

![alt text](notes_images/duelling.png)

## ["Rainbow: Combining Improvements in Deep Reinforcement Learning", 2018](https://arxiv.org/abs/1710.02298)

"This paper examines six extensions to the DQN algorithm and empirically studies their combination."

- Double DQN

- Duelling DQN

- Prioritized Replay

- N-step Returns

- Distributional RL

- Noisy Networks

# Lecture 4.1 - Seminar

In RL we don't use **dropout** or **batch normalization**.

# Lecture 5 - Distributional RL

## Notation

["A Distributional Perspective on Reinforcement Learning", 2017](https://arxiv.org/abs/1707.06887)

$$
G_t = \sum_{t' = t}^{\infty} \gamma^{t' - t} r_{t'} \quad \text{\small\color{gray}Random variable}
$$

$$
V^{\pi}(s) = \mathbb{E}_{\pi} \left[ G_t \mid s_t = s \right] \quad \text{\small\color{gray}Number}
$$

$$
Q^{\pi}(s, a) = \mathbb{E}_{\pi} \left[ G_t \mid s_t = s, a_t = a \right] \quad \text{\small\color{gray}Number}
$$

$$
Z^{\pi}(s, a) = \left[ G_t \mid s_t = s, a_t = a \right] \quad \text{\small\color{gray}Random variable}
$$

Therefore:

$$
Q^{\pi}(s, a) = \mathbb{E}[Z^{\pi}(s, a)].
$$

> So instead of modeling only its expectation $Q^{\pi}$, we model the full distribution $Z^{\pi}$ of returns.

### Recurrent Relation

$$
Z^{\pi}(x, a) \overset{D}{=} R(x, a) + \gamma Z^{\pi}(X', A')
$$

### Bellman Operator

$$
\mathcal{T}Z(x, a) \overset{D}{=} R(x, a) + \gamma Z(X', \arg\max_{a' \in \mathcal{A}} \mathbb{E} Z(X', a'))
$$

> Instead of mapping distributions to expectations at every step, we propagate distributions themselves

## Probability Approximation: C51 vs QR-DQN

| Aspect                         | **C51 (Categorical DQN)**                                                                                                                                        | **QR-DQN (Quantile Regression DQN)**                                                                                                                                                                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Representation of $Z(s,a)$** | $ Z_\theta(s,a) = \sum_{i=1}^N p_i(s,a)\,\delta_{z_i} $ where $\{z_i\}$ are **fixed atoms**, $p_i$ are softmax probabilities.                  | $ Z_\theta(s,a) = \frac{1}{N} \sum_{i=1}^N \delta_{\theta_i(s,a)}$ where $\theta_i(s,a)$ are **learned quantile values**, all equally weighted.                                                                                             |
| **Atoms / Support**            | Fixed grid:$ z_i = V_{\min} + (i-1)\Delta z,\quad \Delta z = \frac{V_{\max}-V_{\min}}{N-1}$.                                                  | Learned per state-action: $\theta_i(s,a)$ directly parameterized by NN.                                                                                                                                                                                       |
| **Probabilities**              | Learned: $p_i(s,a) = \mathrm{softmax}(h_\theta(s,a))$.                                                                                                           | Fixed: $1/N$ for each quantile.                                                                                                                                                                                                                               |
| **Expectation recovery**       | $ Q(s,a) = \sum_{i=1}^N p_i(s,a)\,z_i.$                                                                                                             | $ Q(s,a) = \frac{1}{N} \sum_{i=1}^N \theta_i(s,a).$                                                                                                                                                                                              |
| **Target distribution**        | $ r + \gamma Z(x', a^*), \quad a^* = \arg\max_{a'} \sum_i p_i(x',a')\,z_i.$                                                                         | $ y_j = r + \gamma \theta_j(x',a^*), \quad a^* = \arg\max_{a'} \frac{1}{N} \sum_{j=1}^N \theta_j(x',a').$                                                                                                                                        |
| **Projection step**            | Required: project shifted atoms $r+\gamma z_i$ back to fixed support $\{z_i\}$.                                                                                  | Not required: quantiles move freely (no projection).                                                                                                                                                                                                          |
| **Loss function**              | Cross-entropy (KL divergence) between predicted categorical dist. and projected target dist.| Quantile regression loss (pinball loss) |
| **Hyperparameters**            | Need $V_{\min}, V_{\max}, N$.                                                                                                                                    | Only $N$ (number of quantiles).                                                                                                                                                                                                                               |
| **Distribution family**        | Categorical distribution over fixed bins.                                                                                                                        | Empirical quantile distribution (flexible, unbounded).                                                                                                                                                                                                        |

# Lecture 6 - Policy Gradient Methods

## Policy Gradient Theorem

### 1. Setup

$$
\pi_\theta(a|s) = \Pr[a_t = a \mid s_t = s; \theta]
$$

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[ G_0 \right] = \mathbb{E}_{\pi_\theta}\!\left[ \sum_{t=0}^\infty \gamma^t r(s_t,a_t) \right]
$$

- Objective: $J(\theta) \to \max$

### 2. Problem

Naively differentiating through the expectation is tricky because the trajectory distribution depends on $\pi_\theta$.

### 3. Log-Derivative Trick

$$
\nabla_\theta \log \pi_\theta(a|s) = \nabla_\theta \pi_\theta(a|s) \cdot \frac{1}{\pi_\theta(a|s)}
$$

$\to$ trick:

$$
\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)
$$

This allows gradients to pass inside probabilities.

### 4. Distribution of Trajectories

Trajectory $\tau = (s_0,a_0,s_1,a_1,\ldots)$ has probability:

$$
p_\theta(\tau) = \rho(s_0) \prod_{t=0}^\infty \pi_\theta(a_t|s_t)\, p(s_{t+1}|s_t,a_t)
$$

where $\rho(s_0)$ is initial state distribution.  
Thus,

$$
J(\theta) = \int p_\theta(\tau) R(\tau) \, d\tau
$$

with $R(\tau) = \sum_{t=0}^\infty \gamma^t r(s_t,a_t)$.

### 5. Gradient of Objective

$$
\nabla_\theta J(\theta) =
$$
$$
= \int \nabla_\theta p_\theta(\tau) R(\tau) \, d\tau =
$$

$$
= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) \, d\tau =
$$

$$
= \mathbb{E}_{\tau \sim p_\theta}\! \left[ R(\tau)\, \nabla_\theta \log p_\theta(\tau) \right] =
$$

$$
= \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty R(\tau) \, \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$

But $R(\tau)$ is the *full return*, not aligned with timestep $t$.

### 6. Step-Wise Return Decomposition

$$
R(\tau) = G_t + \text{(rewards before } t)
$$

Since rewards before $t$ are independent of $a_t$, we can replace $R(\tau)$ with **return from $t$:**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty G_t \, \nabla_\theta \log \pi_\theta(a_t|s_t) \right]
$$

### 7. Policy Gradient Theorem

The theorem states:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta} \left[ Q^{\pi_\theta}(s,a) \, \nabla_\theta \log \pi_\theta(a|s) \right]
$$

where:

- $d^{\pi_\theta}(s)$ = discounted state distribution under policy $\pi_\theta$,
- $Q^{\pi_\theta}(s,a)$ = expected return starting from state $s$, action $a$.

## REINFORCE

Since $ Q^{\pi}(s_t, a_t) = \mathbb{E}\!\left[ G_t \mid s_t, a_t \right] $, we can replace it with the sampled return $ G_t $ (Monte Carlo estimate):

$$
\nabla_{\theta} J(\theta)
\;\approx\;
\mathbb{E}_{\pi_{\theta}}
\!\left[
\sum_{t=0}^{\infty}
\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\; G_t
\right].
$$

Update rule:

$$
\theta \;\leftarrow\; \theta \;+\; \alpha \sum_{t=0}^{T}
\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\; G_t .
$$

> # 47:00

> # Other Lectures

# [MIT 6.S191 - Reinforcement Learning](https://www.youtube.com/watch?v=to-lHJfK4pw)

### 0. Model-Based Learning

- **Learns**: a model of the environment — transition probabilities $ P(s'|s, a) $ and reward function $ R(s, a) $
- **Examples**:
  - Dyna-Q
  - MuZero

### 1. Value Learning

- **Learns**: $ V(s) $ or $ Q(s, a) $
- **Examples**:
  - Q-learning
  - TD-learning
  - SARSA
- **Policy**: derived from values, policy strategies:
  - Greedy
  - $\epsilon$-greedy
  - Softmax / Boltzmann policy
  - Greedy in the limit of infinite exploration (GLIE)
  - Stochastic policies

### 2. Policy Learning

- **Learns**: policy $ \pi(a|s) $ directly
- **Examples**:
  - REINFORCE
  - PPO (Proximal Policy Optimization)
  - TRPO (Trust Region Policy Optimization)
- **Policy**: explicitly parameterized and optimized

### 3. Actor-Critic Learning

- **Learns**: both a policy (**actor**) and a value function (**critic**)
- **Examples**:
  - A2C, A3C (Advantage Actor-Critic)
  - PPO (also has actor-critic form)
  - DDPG, TD3, SAC (in continuous control)

### 4. Imitation Learning

- **Learns**: from expert demonstrations
- **Examples**:
  - Behavior Cloning
  - GAIL (Generative Adversarial Imitation Learning)

### 5. Inverse Reinforcement Learning (IRL)

- **Learns**: the reward function from expert behavior
- **Goal**: explain why expert acts that way by inferring $ R(s, a) $

![alt text](notes_images/deep_rl_algorithms.png)

## 1. Deep Q Networks

![alt text](notes_images/q_network.png)

> Use when action space is small & _discrete_.

## 2. Deep Policy Networks / Policy Learning

![alt text](notes_images/policy_network.png)

> Use when action space is _continuous_ and you need to model stochastic policies.

## 3. Actor-Critic

In sparse-reward games like Chess or Go, you only get a reward at the end (e.g., win = +1, lose = -1) $\to$ Pure policy gradient is high variance — reward signal is too delayed.

- **Actor**: policy network $\pi_\theta(a \mid s)$ — decides what to do.

- **Critic**: value function $V_w(s)$ (or $Q(s,a)$) — evaluates how good the actor’s decision was.

# [MIT 6.S191 - (Google) Large Language Models](https://www.youtube.com/watch?v=ZNodOsz94cc)

"""
Prompt: You're ...\n
User: ...\n
ChatBot:
"""

**Few/One/Zero-shot** example. No gradient updates are performed.

Standrad Prompting $\to$ **Chain-of-thought prompting**: "let's think step by step".

![alt text](notes_images/summary_of_approaches.png)

# [Daniel Han - Full Workshop](https://www.youtube.com/watch?v=OkEGJ5G3foU)

## PPO

**PPO** constrains the **policy ratio** to keep it within a **trust region** $[1 - \epsilon,\ 1 + \epsilon]$ on policy update while training.

$$
J(\theta) = \mathbb{E} \left[ \min \left( r_t A_t,\ \text{clip}(r_t,\ 1 - \epsilon,\ 1 + \epsilon) A_t \right) \right]
$$

- $ r*t = \frac{π*\theta(a*t \mid s_t)}{π*{\text{old}}(a_t \mid s_t)} $

## GRPO

**GRPO** = PPO + Group-based normalization

GRPO computes advantages within each group of all possible actions and normalizes them:

$$
\tilde{A}_{i,t} = \frac{A_{i,t} - \mu_i}{\sigma_i}
$$

Where:

- $i$ is the group index,
- $\mu_i = \mathbb{E}[A_{i,t}]$ is the mean advantage within group $i$,
- $\sigma_i = \sqrt{\mathbb{V}[A_{i,t}]}$ is the standard deviation within group $i$.

This normalization is done _independently per group_.

$$
J(\theta) = \frac{1}{G} \sum_{i=1}^{G} \min \left( r_i \tilde{A}_{i,t},\ \text{clip}(r_i,\ 1 - \varepsilon,\ 1 + \varepsilon) \tilde{A}_{i,t} \right)
$$

- $G$ is the number of groups

## Training path

![alt text](notes_images/training_path.png)

- SFT — **Supervised Fine-Tuning**

- IFT — **Instruction Fine-Tuning**

- Pref FT — **Preference Fine-Tuning**: model is trained to prefer the higher-ranked response — not by supervised labels, but by optimizing over _rankings_

  ❌ Output A: “Sure, here’s how to build a bomb...”

  ✅ Output B: “Sorry, I can’t help with that request.”

- RLVR — **Reinforcement Learning with Verifiable Rewards**: automated, rule-based, or formalized feedback signals

- RLHF (**Reinforcement Learning from Human Feedback**)

> Use `torch.compile`.
