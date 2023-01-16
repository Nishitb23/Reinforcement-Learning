import torch
import torch.nn as nn
import algorithm.DQN.training as tr
from algorithm.DQN.replay_buffer import buffer

class DQNnet(nn.Module):

    def __init__(self, state_length, n_actions):
        super(DQNnet, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_length, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
            )

    def forward(self, x):
        return self.model(x)


class DQNagent():
    def __init__(self, env_model, device= None) -> None:

        self.policy_net = DQNnet(env_model.state_length,len(env_model.action_list))
        self.target_net = DQNnet(env_model.state_length,len(env_model.action_list))
        self.memory = buffer()
        self.device = device
        self.env_model = env_model
        self.target_net.load_state_dict(self.policy_net.state_dict())
            

    def train(self):
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            num_episodes = 600
        else:
            num_episodes = 10
            self.device = torch.device("cpu")

        tr.train(self.target_net, self.policy_net, self.env_model, self.memory, num_episodes, self.device)
        torch.save(self.policy_net.state_dict(), "algorithm/DQN/saved_models/policy_net")
        torch.save(self.target_net.state_dict(), "algorithm/DQN/saved_models/target_net")

    def step(self):
        state = torch.tensor(self.env_model.grid.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.policy_net(state).max(1)[1][0])
        #return self.policy_net(self.env_model.grid.flatten()).max(1)[1].view(1, 1)




        