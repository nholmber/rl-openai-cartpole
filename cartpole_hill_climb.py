import gym
import time
import numpy as np
import matplotlib.pyplot as plt

MAX_ITER = 500

def get_model(distribution=None):
    if distribution is None:
        return 2*np.random.rand(4)-1.
    else:
        return distribution

def run_episode(env, model, num_steps, render=False):
    observation = env.reset()
    for t in range(num_steps):
        if render: env.render()
        # Action is type Discrete(2): 0 moves cart left, 1 right
        action = 0 if np.dot(model, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        # Slow down framerate
        if render: time.sleep(0.01)
        if done:
            break

    return t+1

def run_iteration(env, num_episodes, num_steps, noise_scale):
    # Return model which achieves perfect reward
    reward = 0
    max_reward = num_episodes*num_steps

    # Initialize random model
    model = get_model()
    num_iterations = 0

    while reward < max_reward:
        model_reward = 0
        if num_iterations > 0:
            # Add noise to model weights
            attempt = model + noise_scale * get_model()
        else:
            attempt = model

        for _ in range(num_episodes):
            model_reward += run_episode(env, attempt, num_steps)

        if model_reward > reward:
            reward = model_reward
            model = attempt

        num_iterations += 1
        if num_iterations > MAX_ITER:
            print('Warning iteration did not converge.')
            break

    return num_iterations

def main():
    # Note: gym.make('') imposes a 200 limit in the number steps
    # Limit can be bypassed by calling gym.make('').env
    env = gym.make('CartPole-v0')
    num_episodes = 1
    num_steps = 200
    num_iter = 50
    noise_scale = .2

    trials = np.zeros(num_iter)
    for i in range(num_iter):
        trials[i] = run_iteration(env, num_episodes, num_steps, noise_scale)

    plt.style.use('seaborn')
    plt.figure()
    plt.hist(trials, edgecolor='black', linewidth=1.2)
    converged = trials[np.where(trials < MAX_ITER)]
    print('Average number of episodes needed to reach convergence: {}'.format(converged.mean()))
    plt.show()


if __name__ == '__main__':
    main()