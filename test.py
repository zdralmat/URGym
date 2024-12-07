from urgym.envs.env_two_balls_balance_v0 import TwoBallsBalance
import numpy as np
import time

# Create the environment
env = TwoBallsBalance()
env.reset()
actions = env.action_space
print("quii:",actions)
input("Press Enter to continue...")
for i in range(10000):
    action = np.random.uniform(-0.1, 0.1, 3)
    print(action)
    #lol = env.step(action)
    #print(lol)
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f'Observation: {obs}')
    print(f'Reward {reward}')
    print(f'Terminated: {terminated}')
    time.sleep(0.1)

env.close()
