# iLunarLander
Instructing an Actor-Critic Agent to Land on the Moon


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


REIFORCE 
