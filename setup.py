from env.env_model import model
import numpy as np
import sys
from algorithm.DQN.DQNnet import DQNagent


grid = np.load('saved_env/env.npy')
#model = model("Maze",grid)
env = model("Maze")

agent = DQNagent(env)
agent.train()

done = False
total_reward = 0
env.reset()
print(env.grid)

while(done!=True):
    env.show()
    #print(env.grid)
    action = agent.step()
    print(action)
    obs,reward,done = env.perform_action(action)
    total_reward += reward
    print("hey")

print("total reward: ",total_reward)