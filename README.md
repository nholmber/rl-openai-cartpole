# Cartpole RL
My implementations of reinforcement learning algorithms for solving [CartPole](https://openai.com/requests-for-research/#cartpole) in [OpenAI Gym](https://gym.openai.com/docs/).

Algorithms implemented:
* `cartpole_policy_gradient.py`: [Vanilla Policy Gradient algorithm](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#) using a stochastic MLP policy implemented with Tensorflow. Variants:
	* Default reward method: All actions given full reward
	* Rewards-to-go method: Actions given only future reward
	* On-policy value baseline: Uses value function as baseline

<img src="https://github.com/nholmber/rl-openai-cartpole/blob/master/img/policy.png?raw=true" width="600" title="Policy Gradient Training">

* `cartpole_hill_climb.py`: Linear policy where weights are update with random noise
* `cartpole_random.py`: Linear policy where weights are drawn randomly
