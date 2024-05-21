# iLunarLander

<div align="center">
  <img src="iLunarLander.jpg" alt="drawing" width="400"/>
</div>

Guiding intelligent agents to successfully land [LunarLander-v2 of GYM](https://gymnasium.farama.org/environments/box2d/lunar_lander/) on the moon through the implementation of REINFORCE and Actor-Critic algorithms! Enjoy the journey! 😄

## REINFORCE

### Objetive
This sub-repository delves into the utilization of Policy Gradient (PG) for training of the corresponding intelligent.

### Approach
PG represents an approach to tackling Reinforcement Learning (RL) challenges, striving to discover an optimal behavioral strategy (or policy) for the agent to acquire maximum rewards. PG methods focus on modeling and enhancing the policy directly from the probability distribution of actions, contrasting with Q-learning where the agent selects the best action directly based on state-action values.

The policy is typically represented by a parameterized function with respect to $\mathbf{\theta}$, denoted as $\pi_{\mathbf{\theta}}(a|s)$, where $a$ and $s$ represent the action and the state, respectively. $\pi_{\mathbf{\theta}}(a|s) = \mathcal{P} \lbrace A_{t} = a | S_{t} = s \rbrace$ signifies the probability of an action $a$ at time step $t$, given the state $s$ at timestep $t$ and the parameters $\mathbf{\theta}$ of the policy.

Now if we ensure that $\pi$ is a valid probability distribution with respect to ${\mathbf{\theta}}$, we can then define a performance measure function $J(\mathbf{\theta})$ and employ gradient ascent to adjust $\mathbf{\theta}$ in order to discover the optimal policy: $\mathbf{\theta}\_{t+1} = \mathbf{\theta}\_{t} + \alpha \nabla J(\mathbf{\theta}\_{t})$.

- Ensuring probabilistic validity often involves feeding the values of each state-action pair ($s, a$) (such as those generated by a neural network with parameters $\mathbf{\theta}$ after receiving a state) into a softmax function, where the result is denoted by $h(s, a, \mathbf{\theta})$. This process guarantees that $\pi_{\mathbf{\theta}}(a|s) \in (0, 1)$, as demonstrated below:

  $$\pi_{\mathbf{\theta}}(a|s) = \frac{\exp h(s, a, \mathbf{\theta})}{\sum_{a' \in \mathbf{\mathcal{A}}} \exp h(s, a', \mathbf{\theta})}$$

  By following this approach, action preferences enable the agent to adopt a deterministic policy, shaping a probability distribution. Consequently, the likelihood of the best action tends toward $1$. Another benefit is the smooth adjustment of action probabilities across a function of the policy parameter. This mitigates issues such as overestimation of the importance of a selected action, thus avoiding convergence into a suboptimal policy, a common pitfall in methods like epsilon greedy.

- Now, let's delve into defining $J(\mathbf{\theta})$ and computing its gradient. Sutton & Barto, in [this link](http://incompleteideas.net/book/bookdraft2017nov5.pdf) (also elaborated [here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)), tackled this issue and demonstrated that the gradient adheres to the following equation:

  $$\nabla J(\mathbf{\theta}) = \sum_{s \in \mathbf{\mathcal{S}}} \mu(s) \left ( \sum_{a \in \mathbf{\mathcal{A}}} q^{\pi}(s, a) \nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|s) \right ),$$
  
  where $\mu(s)$ represents the probability of being in state $s$ according to our stochastic policy $\pi$, while $q$ denotes an action-value function aligned with this policy.

### Algorithm

Now equipped with the policy gradient equation, we can devise a straightforward algorithm leveraging gradient ascent to adjust our policy parameters. While the theorem encompasses a summation over all states and actions, given the impracticality of computing gradients for all potential states and actions, we can rely on a sampled gradient. This method is known as REINFORCE (Monte-Carlo policy gradient), introduced by Sutton & Barto, which relies on estimated returns obtained through Monte-Carlo methods using episode samples. REINFORCE proves effective because the expectation of the sample gradient aligns with the actual gradient. In essence:

$$\begin{eqnarray} 
\nabla J(\mathbf{\theta}) &=& \sum_{s \in \mathbf{\mathcal{S}}} \mu(s) \left ( \sum_{a \in \mathbf{\mathcal{A}}} q^{\pi}(s, a) \nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|s) \right ) \nonumber \\
&=& \mathbb{E}\_{s \sim \pi} \left [ \sum_{a \in \mathbf{\mathcal{A}}} q^{\pi}(S_{t}, a) \nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|S_{t}) \right ] \nonumber \\
&=& \mathbb{E}\_{s \sim \pi} \left [ \sum_{a \in \mathbf{\mathcal{A}}} \pi_{\mathbf{\theta}}(a|S_{t}) q^{\pi}(S_{t}, a) \frac{\nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a|S_{t})}{\pi_{\mathbf{\theta}}(a|S_{t})} \right ] \nonumber \\
&=& \mathbb{E}\_{s, a \sim \pi} \left [ q^{\pi}(S_{t}, A_{t}) \frac{\nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(A_{t} | S_{t})}{\pi_{\mathbf{\theta}}(A_{t} | S_{t})} \right ] \nonumber \\
&=& \mathbb{E}\_{s, a \sim \pi} \left [ q^{\pi}(S_{t}, A_{t}) \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(A_{t} | S_{t}) \right ] \qquad \text{because } (\ln x)^{'} = \frac{1}{x} \nonumber \\
&=& \mathbb{E}\_{s, a \sim \pi} \left [ G_t \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(A_{t} | S_{t}) \right ] \qquad \text{because } q^{\pi}(S_{t}, A_{t}) = \mathbb{E}\_{s, a \sim \pi} \left [ G_t | S_{t}, A_{t} \right ] \nonumber \\
\end{eqnarray}$$

Hence, we can compute $G_t$ from actual sample trajectories and utilize it to adjust our policy gradient. This approach hinges on complete trajectories, hence earning its classification as a Monte-Carlo method. The procedure is rather straightforward:

1. Initialize the policy parameter $\mathbf{\theta}$ at random.
  
2. Generate one trajectory on policy $\pi_{\mathbf{\theta}}$: $S_1, A_1, R_2, S_2, A_2, ..., S_T$.

3. For $t = 1, 2, ..., T$:
   
   a. Estimate the the return $G_t$;
  
   b. Update policy parameters: $\mathbf{\theta} \gets \mathbf{\theta} + \alpha \gamma_t G_t \nabla_{\mathbf{\theta}} \ln \pi_{\mathbf{\theta}}(A_{t} | S_{t})$

This process is implemented in [main.py](REINFORCE/main.py). $G_t$ is estimated using a Deep Neural Network (DNN) in [PGN.py](REINFORCE/PGN.py), then processed through softmax in [PG_Agent.py](REINFORCE/PG_Agent.py). The resulting per-action probabilities are fed into the Categorical distribution for action selection. The categorical distribution is a discrete probability distribution used to model scenarios where there are a fixed number of possible outcomes, each with an associated probability. It's commonly employed in reinforcement learning to select actions from a set of discrete choices. A fundamental function of the categorical distribution is sampling, which involves randomly selecting an outcome based on its associated probability.

### Getting Started

Ensure that you've installed all the packages listed in [requirements.txt](REINFORCE/requirements.txt) and execute [main.py](REINFORCE/main.py). The resulting figures will be saved in [plots](REINFORCE/plots/). Moreover, you can observe the lunar lander and its operations under the agent's control using `env.render()`. For a sample code, refer to [test.py](REINFORCE/test.py).


### Outcome

The average scores (total rewards accumulated) of the lunar lander over 1000 training steps:

<div align="center">
  <img src="REINFORCE/plots/REINFORCE_LunarLander-v2_0.0005_1000.png" alt="drawing" width="400"/>
</div>
