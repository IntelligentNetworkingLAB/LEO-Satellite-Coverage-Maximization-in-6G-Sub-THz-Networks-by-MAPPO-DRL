import numpy as np
import gym
from gym import spaces
import environment
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0.0, high=30.0, shape=(8,), dtype=np.float32)
        self.env = environment.EnvironmentRoute()
    def step(self, action):
        self.env.set_action_route(action)
        observation, reward, done, info = self.env.get_step_route()
        self.env.step_route()
        return np.array(observation, dtype=np.float32), reward, done, info
    def reset(self):
        observation = self.env.reset()
        return np.array(observation, dtype=np.float32)
    def close (self):
        print("close")

# Main 함수
if __name__ == '__main__':
    env = CustomEnv()
    env = Monitor(env)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
    )
    eval_callback = EvalCallback(env, eval_freq=1000, deterministic=True, render=False)
    model.learn(total_timesteps=50000, callback=[eval_callback])
    model.save("./stable_baseline3/models/ppo_v2")