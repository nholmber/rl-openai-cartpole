import gym
import time
import numpy as np

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

def main():
    # Note: gym.make('') imposes a 200 limit in the number steps
    # Limit can be bypassed by calling gym.make('').env
    env = gym.make('CartPole-v0')
    num_episodes = 5
    num_steps = 200
    num_models  = 1000

    # Return model with largest cumulative reward
    best_reward = 0
    best_model = None

    for _ in range(num_models):
        model = get_model()
        model_reward = 0
        for i_episode in range(num_episodes):
            model_reward += run_episode(env, model, num_steps)

        if model_reward > best_reward:
            best_reward = model_reward
            best_model = model

    print('Best reward {} out of max {}, model: {}'.format(best_reward, num_episodes*num_steps, best_model))

    # Render one episode with best model
    model = best_model
    _ = run_episode(env, best_model, num_steps, render=True)
    env.close()

if __name__ == '__main__':
    main()