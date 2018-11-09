import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import gym
import time

class VPG:
    """Vanilla Policy Gradient algorithm"""
    def __init__(self, num_actions, hidden_sizes, observation_size, lr=0.01, activation=tf.tanh, reward_method='total',
                 value_baseline=False):
        self._num_actions = num_actions
        self._hidden_sizes = hidden_sizes
        self._observation_size = observation_size
        self._lr = lr
        self._activation = activation
        self._reward_method = reward_method
        self.use_value = value_baseline
        self.policy_network()
        if self.use_value: self.value_network()

    @property
    def use_value(self):
        return self._use_value

    @use_value.setter
    def use_value(self, bool):
        self._use_value = bool

    def policy_network(self):
        # Placeholder for observation from Gym
        self._obs_ph = tf.placeholder(shape=(None, self._observation_size), dtype=tf.float32, name='observations')

        # Placeholder tensors for rewards/actions obtained during epoch
        self._rewards_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='rewards')
        self._actions_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='actions')

        # Stochastic policy is modeled with a MLP
        logits = mlp(self._obs_ph, self._hidden_sizes+[self._num_actions], activation=self._activation)

        # Select action by sampling
        self._action = tf.squeeze(tf.multinomial(logits, 1), axis=1)

        # Log likelihoods for all actions
        log_probs_all = tf.nn.log_softmax(logits)

        # One hot vectors representing the actions taken during epoch
        selected_actions = tf.one_hot(self._actions_ph, self._num_actions)

        # Log likelihoods for all executed actions
        log_probs_selected = tf.reduce_sum(selected_actions * log_probs_all, axis=1)

        # Train model by maximizing total undiscounted future reward R
        # that is minimize -R
        self._loss = -tf.reduce_mean(self._rewards_ph * log_probs_selected)
        self._optimizer = tf.train.AdamOptimizer(self._lr).minimize(self._loss)

    def value_network(self):
        # Use value function as baseline in policy gradient
        # Value functions is also modeled with an MLP
        self._values = tf.squeeze(mlp(self._obs_ph, self._hidden_sizes+[1], activation=self._activation), axis=1)

        # Train network by minimizing MSE error between predicted values and experienced rewards
        self._value_loss = tf.losses.mean_squared_error(self._rewards_ph, self._values)
        self._value_optimizer = tf.train.AdamOptimizer(self._lr).minimize(self._value_loss)

    def predict(self, sess, observation):
        # Forward pass observation through MLP
        return sess.run(self._action, feed_dict={self._obs_ph: observation})

    def policy_update(self, sess, observations, actions, rewards):
        # Optimize MLP weights using backprop
        opt = [self._loss, self._optimizer]

        # Reshape data
        rewards = np.concatenate([np.array(i) for i in rewards])
        observations = np.array(observations)

        # Compute state-value function for current policy and subtract it from the rewards
        if self.use_value:
            rewards -= sess.run(self._values, feed_dict={self._obs_ph: observations})

        feed_dict = {self._rewards_ph: rewards,
                     self._actions_ph: np.array(actions),
                     self._obs_ph: observations}

        return sess.run(opt, feed_dict=feed_dict)

    def value_update(self, sess, observations, rewards):
        if not self.use_value: return

        # Optimize value network weights
        opt = [self._values, self._value_loss, self._value_optimizer]

        # Reshape data
        rewards = np.concatenate([np.array(i) for i in rewards])
        observations = np.array(observations)

        feed_dict = {self._rewards_ph: rewards,
                     self._obs_ph: observations}

        _, loss, _ = sess.run(opt, feed_dict=feed_dict)
        return loss

    def compute_reward(self, reward_list):
        # Compute rewards along epoch
        if self._reward_method == 'total':
            # Assign each action the full undiscounted return from the whole episode
            return [sum(reward_list)]*len(reward_list)

        elif self._reward_method == 'to-go':
            # Reward-to-go i.e. each action gets only rewards from future states
            reward = [0]*len(reward_list)
            for i in reversed(range(len(reward_list))):
                reward[i] = reward_list[i]
                if i < len(reward_list)-1: reward[i] += reward[i+1]
            return reward

def mlp(x, sizes, activation=tf.tanh):
    # Returns densely connected NN with len(sizes) layers
    # Note: Output layer has no activation
    for i, size in enumerate(sizes[:-1]):
        x = Dense(size, activation=activation)(x)

    return Dense(sizes[-1])(x)


def run_epoch(sess, env, vpg, max_iterations_per_epoch, num_steps, render=False, num_value_iter=80):
    epoch_completed = False
    num_iterations = 0

    # Store actions, rewards, observations in this epoch
    epoch_rewards = []
    epoch_actions = []
    epoch_observations = []
    average_reward = []
    average_episode_len = []

    while not epoch_completed:
        # Reset episode specific variables
        observation, episode_reward = env.reset(), []

        # Start new episode
        for i in range(num_steps):
            if render: env.render()

            # Store observations
            epoch_observations.append(observation.copy())

            # Compute next action based on current policy
            action = vpg.predict(sess, observation.reshape(1, -1))

            # Perform action
            observation, reward, done, _ = env.step(action[0])

            # Accumulate action and reward histories
            epoch_actions.append(action[0])
            episode_reward.append(reward)

            # Exit if simulation fails or number of steps reaches limit
            if done or i == num_steps-1:
                # Assign reward for each selected actions
                epoch_rewards.append(vpg.compute_reward(episode_reward))

                # Statistics
                average_reward.append(sum(episode_reward))
                average_episode_len.append(i+1)

                # Stop accumulating experience in this epoch if max iteration count is reached
                num_iterations += i
                if num_iterations > max_iterations_per_epoch: epoch_completed = True
                break

        if render: render = False

    # Update policy network
    loss, _ = vpg.policy_update(sess, epoch_observations, epoch_actions, epoch_rewards)

    # Update value network
    for _ in range(num_value_iter):
        value_loss = vpg.value_update(sess, epoch_observations, epoch_rewards)

    # Return statistics from epoch
    return loss, np.mean(average_reward), np.mean(average_episode_len)


def main(args):
    sess = tf.Session()
    env = gym.make('CartPole-v0')

    # For reproducible training during debugging
    if args.fixed_seed:
        tf.set_random_seed(0)
        env.seed(0)

    hidden_sizes = [32, 32]
    num_actions = env.action_space.n
    observation_size = env.observation_space.shape[0]
    vpg = VPG(num_actions, hidden_sizes, observation_size, activation=tf.nn.relu, reward_method='to-go', value_baseline=True)
    sess.run(tf.global_variables_initializer())

    num_epochs = 50
    num_steps = 200
    max_iterations_per_epoch = 4000
    render = args.render

    for i in range(num_epochs):
        loss, reward, length = run_epoch(sess, env, vpg, max_iterations_per_epoch, num_steps, render=render)
        print('Epoch {}: Loss: {:.3f}, Average Reward {:.3f}, Average Episode Length {:.3f}'.format(
              i, loss, reward, length))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', help='Render environment?', action='store_true')
    parser.add_argument('--fixed-seed', help='Use a fixed seed to initialize Gym and Tensorflow?', action='store_true')
    args = parser.parse_args()
    main(args)