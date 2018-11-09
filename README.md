# Cartpole RL
My implementations of reinforcement learning algorithms for solving [CartPole](https://openai.com/requests-for-research/#cartpole) in [OpenAI Gym](https://gym.openai.com/docs/).

Algorithms implemented:
* `cartpole_policy_gradient.py`: [Vanilla Policy Gradient algorithm](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#) using a stochastic MLP policy implemented with Tensorflow
* `cartpole_hill_climb.py`: Linear policy where weights are update with random noise
* `cartpole_random.py`: Linear policy where weights are drawn randomly
