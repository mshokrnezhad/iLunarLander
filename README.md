# iLunarLander
Instructing intelligent agents to land on the moon utilizing REINFORCE and Acotr-Critic algorithms! Have fun :)

## REINFORCE
This sub-repository explores the application of Policy Gradient (PG) to train an intelligent agent to successfully land a lunar lander on the moon's surface.

PG is an approach to solve Reinforcement Learning (RL) problems, with the aim of finding an optimal behavior strategy (or policy) for the agent to obtain optimal rewards. The PG methods target at modeling and optimizing the policy directly form the probability distribution of actions (unlike Q learning where the agent selects the best action based on state-action values).

The policy is usually modeled with a parameterized function respect to $\mathbf{\theta}$, $\pi_{\mathbf{\theta}}(a|s)$, where $a$ and $s$ represent the action and the state, respectively. $\pi_{\mathbf{\theta}}(a|s) = \mathcal{P} \lbrace A_{t} = a | S_{t} = s \rbrace$, which is the probability of an action $a$ at time step $t$ given the state $s$ at timestep $t$ and the policy’s parameters $\mathbf{\theta}$.

Now if we can assure that $\pi$ is a valid probability distribution with respect to ${\mathbf{\theta}}$, we can define a performance measure function $J(\mathbf{\theta})$ and use gradient ascent to adjust $\mathbf{\theta}$ to find the optimal policy: $\theta_{t + 1} = \mathbf{\theta}_{t} + \alpha \nabla J(\mathbf{\theta}_{t})$.

To ensure the probabilistic validity, a promissing way is to feed the values of each state-action pair ($s, a$) (such as the values produced by a neural network with parameters $\mathbf{\theta}$ after receiving an state), dubbed $h(s, a, \mathbf{\theta})$, into softmax ensuring that $\pi_{\mathbf{\theta}}(a|s) \in (0, 1)$, as follows:

$$
\pi_{\mathbf{\theta}}(a|s) = \frac{\exp h(s, a, \mathbf{\theta})}{\sum_{a' \in \mathbf{\mathcal{A}}} \exp h(s, a', \mathbf{\theta})}
$$

Doing so, action preferences allow the agent to approach a deterministic policy, forming a probability distribution. This means that the probability of the best action will be driven to approach 1. Another advantage is that the action probability is adjusted smoothly over a function of the policy parameter, which solves the problem of overestimation of the importance of a selected action and converge into a suboptimal policy in approaches such as epsilon greedy.

Now, it is time to define $J(\mathbf{\theta})$ and calculate its gradient.Sutton & Barto in [this link](http://incompleteideas.net/book/bookdraft2017nov5.pdf) (also explained [here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)) worked on this problema and proved that the gradient follows the follwoing equation:

$$
\nabla J(\mathbf{\theta}) \equiv  \sum_{s \in \mathbf{\mathcal{S}}} \mu(s) \bbig( \sum_{a \in \mathbf{\mathcal{A}}} q^{\pi}(s, a) \bbig)
$$



The value of the reward (objective) function depends on this policy and then various algorithms can be applied to optimize e for the best reward. The reward function is defined as: 

$$
J(\theta) = \sum_{s \in \mathbf{\mathcal{S}}} d^{\pi}(s)V^{\pi}(s) = \sum_{s \in \mathbf{\mathcal{S}}} \Big( d^{\pi}(s) \sum_{a \in \mathbf{\mathcal{A}}} \pi_{\theta}(a|s) Q^{\pi}(s, a) \Big)
$$

where $d^{\pi}(s)$ is the stationary distribution of Markov chain for $\pi_{\theta}$ (on-policy state distribution under $\pi$). 

Imagine that you can travel along the Markov chain's states forever, and eventually, as the time progresses, the probability of you ending up with one state becomes unchanged this is the stationary probability for $\pi_{\theta}$. d”(s) = limt→∞ P(st = s|80, πθ) is the probability that st = s when starting from so and following policy πe for t steps. Actually, the existence of the stationary distribution of Markov chain is one main reason for why PageRank algorithm works. If you want to read more, check this.
It is natural to expect policy-based methods are more useful in the continuous space. Because there is an infinite number of actions and (or) states to estimate the values for and hence value-based approaches are way too expensive computationally in the continuous space. For example, in generalized policy iteration, the policy improvement step arg maxa∈AQ" (s, a) requires a full scan of the action space, suffering from the curse of dimensionality.
Using gradient ascent, we can move 0 toward the direction suggested by the gradient VeJ(0) to find the best θ for πθ that produces the highest return.




**Objective:**

The goal is to develop an agent that can learn through trial and error to control a simulated lunar lander and achieve a soft landing within a designated zone.

**Approach:**

* We will utilize a reinforcement learning algorithm, likely based on Deep Q-Learning (DQN) or a similar technique.
* The agent will receive rewards for actions that bring it closer to a safe landing and penalties for actions that lead to a crash or unstable descent.
* Through repeated training episodes in a simulated lunar landing environment, the agent will learn the optimal control strategy to achieve the desired outcome.

**Technical Stack (examples):**

* Programming Language (Python, potentially with libraries like TensorFlow or PyTorch)
* Reinforcement Learning Framework (OpenAI Gym, Stable Baselines3)
* Visualization Tools (optional: for visualizing the training process)

**Expected Outcomes:**

* A trained agent capable of autonomously landing the lunar lander on the moon with a high success rate.
* Insights into the effectiveness of reinforcement learning for complex control tasks.
* Open-source code repository to share the implementation and encourage further development.

**Potential Applications:**

* Optimizing control systems for real-world autonomous vehicles (drones, robots)
* Advancing research in artificial intelligence and reinforcement learning
* Educational tool for demonstrating reinforcement learning concepts

**Getting Started:**

This repository provides the source code for the project, allowing you to:

* Set up the development environment
* Train the agent on the lunar landing task
* Visualize the training process (if applicable)
* Experiment with different hyperparameters and learning algorithms

We welcome contributions and discussions to improve the project's capabilities and explore further applications of reinforcement learning.



####################################################################################################################################################################


PGN:
Categorical distribution

The categorical distribution is a discrete probability distribution used to model situations where there are a fixed number of possible outcomes, and each outcome has an associated probability. It's commonly used in reinforcement learning for selecting actions from a set of discrete choices. Here's a breakdown of how it works:

1. Parameters:

The categorical distribution is defined by a single parameter: probabilities for each possible outcome (often called category).
These probabilities represent the likelihood of each outcome occurring.
2. Example:

Imagine you have a 6-sided die. Each side (1, 2, 3, 4, 5, 6) is an outcome in the categorical distribution. In a fair die, each side has an equal probability (1/6).

3. Representing Probabilities:

The probabilities for each outcome can be represented as a vector.
For a die, the probability vector would be: [1/6, 1/6, 1/6, 1/6, 1/6, 1/6].
4. Sampling:

A key function of the categorical distribution is sampling.
Sampling involves randomly selecting an outcome based on its associated probability.
Algorithms like weighted random sampling can be used, where the weight for each outcome is its probability.
5. Applications in Reinforcement Learning:

In reinforcement learning, the categorical distribution is often used to model the agent's policy.
The policy defines the probability of taking each action in a given state.
The neural network in the agent might predict these probabilities for each action.
The act method you saw previously likely samples an action based on these predicted probabilities using the categorical distribution.
6. Key Points:

The categorical distribution is a discrete probability distribution for a fixed number of outcomes.
Each outcome has an associated probability.
Sampling allows selecting an outcome based on its probability.
It's useful in reinforcement learning for modeling the agent's policy (action selection probabilities).


####################################################################################################################################################################


REIFORCE as a policy gradient technique
+
Temporal difference
=
Actor critic

