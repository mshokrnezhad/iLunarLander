# iLunarLander
Instructing intelligent agents to land on the moon utilizing REINFORCE and Acotr-Critic algorithms! Have fun :)

## REINFORCE

### Objetive
This sub-repository explores the application of Policy Gradient (PG) to train an intelligent agent to successfully land a lunar lander on the moon's surface.

### Approach
PG is an approach to solve Reinforcement Learning (RL) problems, with the aim of finding an optimal behavior strategy (or policy) for the agent to obtain optimal rewards. The PG methods target at modeling and optimizing the policy directly form the probability distribution of actions (unlike Q learning where the agent selects the best action based on state-action values).

The policy is usually modeled with a parameterized function respect to $\mathbf{\theta}$, $\pi_{\mathbf{\theta}}(a|s)$, where $a$ and $s$ represent the action and the state, respectively. $\pi_{\mathbf{\theta}}(a|s) = \mathcal{P} \lbrace A_{t} = a | S_{t} = s \rbrace$, which is the probability of an action $a$ at time step $t$ given the state $s$ at timestep $t$ and the policy’s parameters $\mathbf{\theta}$.

Now if we can assure that $\pi$ is a valid probability distribution with respect to ${\mathbf{\theta}}$, we can define a performance measure function $J(\mathbf{\theta})$ and use gradient ascent to adjust $\mathbf{\theta}$ to find the optimal policy: $\mathbf{\theta}\_{t+1} = \mathbf{\theta}\_{t} + \alpha \nabla J(\mathbf{\theta}_{t})$.

- To ensure the probabilistic validity, a promissing way is to feed the values of each state-action pair ($s, a$) (such as the values produced by a neural network with parameters $\mathbf{\theta}$ after receiving an state), dubbed $h(s, a, \mathbf{\theta})$, into softmax ensuring that $\pi_{\mathbf{\theta}}(a|s) \in (0, 1)$, as follows:

  $$\pi_{\mathbf{\theta}}(a|s) = \frac{\exp h(s, a, \mathbf{\theta})}{\sum_{a' \in \mathbf{\mathcal{A}}} \exp h(s, a', \mathbf{\theta})}$$
  
  Doing so, action preferences allow the agent to approach a deterministic policy, forming a probability distribution. This means that the probability of the best action will be driven to approach 1. Another advantage is that the action probability is adjusted smoothly over a function of the policy parameter, which solves the problem of overestimation of the importance of a selected action and converge into a suboptimal policy in approaches such as epsilon greedy.

- Now, it is time to define $J(\mathbf{\theta})$ and calculate its gradient. Sutton & Barto in [this link](http://incompleteideas.net/book/bookdraft2017nov5.pdf) (also explained [here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)) worked on this problema and proved that the gradient follows the follwoing equation:

  $$\nabla J(\mathbf{\theta}) = \sum_{s \in \mathbf{\mathcal{S}}} \mu(s) \big( \sum_{a \in \mathbf{\mathcal{A}}} q^{\pi}(s, a) \nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|s) \big),$$
  
  where $\mu(s)$ is the probability of being at state $s$ following our stochastic policy $\pi$, and $q$ is an action-value function following this policy.

### Algorithm

Now with the policy gradient equation, we can come up with a naive algorithm that makes use of gradient ascent to update our policy parameters. The theorem gives a sum over all states and actions but when we update our parameters, we are only going to be using a sample gradient since there's just no way we can get the gradient for all possible actions and states. The method is called REINFORCE (Monte-Carlo policy gradient), proposed by Sutton & Barto, relying on an estimated return by Monte-Carlo methods using episode samples. REINFORCE works because the expectation of the sample gradient is equal to the actual gradient. In other words:

$$\begin{eqnarray} 
\nabla J(\mathbf{\theta}) &=& \sum_{s \in \mathbf{\mathcal{S}}} \mu(s) \big( \sum_{a \in \mathbf{\mathcal{A}}} q^{\pi}(s, a) \nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|s) \big) \nonumber \\
&=& \mathbb{E}\_{s \sim \pi} \big[ \sum_{a \in \mathbf{\mathcal{A}}} q^{\pi}(S_{t}, a) \nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|S_{t}) \big] \nonumber \\
&=& \mathbb{E}\_{s \sim \pi} \big[ \sum_{a \in \mathbf{\mathcal{A}}} \pi_{\mathbf{\theta}}(a|S_{t}) q^{\pi}(S_{t}, a) \frac{\nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|S_{t})}{\pi_{\mathbf{\theta}}(a|S_{t})} \big] \nonumber \\
&=& \mathbb{E}\_{s, a \sim \pi} \big[ q^{\pi}(S_{t}, A_{t}) \frac{\nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(A_{t} | S_{t})}{\pi_{\mathbf{\theta}}(A_{t} | S_{t})} \big] \nonumber \\
&=& \mathbb{E}\_{s, a \sim \pi} \big[ q^{\pi}(S_{t}, A_{t}) \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(A_{t} | S_{t}) \big] \qquad \text{because } (\ln x)^{'} = \frac{1}{x} \nonumber \\
&=& \mathbb{E}\_{s, a \sim \pi} \big[ G_t \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(A_{t} | S_{t}) \big] \qquad \text{because } q^{\pi}(S_{t}, A_{t}) = \mathbb{E}\_{s, a \sim \pi} \big[ G_t | S_{t}, A_{t} \big] \nonumber \\
\end{eqnarray}$$

Therefore we are able to measure $G_t$ from real sample trajectories and use that to update our policy gradient. It relies on a full trajectory and that’s why it is a Monte-Carlo method.
The process is pretty straightforward:

1. Initialize the policy parameter $\mathbf{\theta}$ at random.
  
2. Generate one trajectory on policy $\pi_{\mathbf{\theta}}$: $S_1, A_1, R_2, S_2, A_2, ..., S_T$.

3. For $t = 1, 2, ..., T$:
   
   a. Estimate the the return $G_t$;
  
   b. Update policy parameters: $\mathbf{\theta} \gets \mathbf{\theta} + \alpha \gamma_t G_t \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(A_{t} | S_{t})$

This porocess is impelemted in [main.py](REINFORCE/main.py). $G_t$ is estimated using a Deep Neural Netwok (DNN) and then pushed into softmax in [PG_Agent.py](REINFORCE/PG_Agent.py). The resulted per action probability is feeded into the Categorical distribution in order to action selection. The categorical distribution is a discrete probability distribution used to model situations where there are a fixed number of possible outcomes, and each outcome has an associated probability. It's commonly used in reinforcement learning for selecting actions from a set of discrete choices. Here's a breakdown of how it works. Imagine you have a 6-sided die. Each side $(1, 2, 3, 4, 5, 6)$ is an outcome in the categorical distribution. In a fair die, each side has an equal probability ($1/6$). A key function of the categorical distribution is sampling. Sampling involves randomly selecting an outcome based on its associated probability.

### Getting Started

### Outcome
