import random
import math
import torch
import numpy as np
#from algorithm.DQN.replay_buffer import buffer


batch_size = 16
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 5000
tau = 0.005
lr = 5e-2
steps_done = 0

#either select random action or action giving higher q-value
def select_action(policy_net, state, actions):

    global steps_done
    sample = random.random()

    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    print("random prob: ",eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            print("followed policy: ",int(policy_net(state).max(1)[1][0]))
            return int(policy_net(state).max(1)[1][0])
            #return policy_net(state).max(1)[1].view(1, 1)
    else:
        return random.sample(actions,1)[0]



def optimize_model(buffer, policy_net, target_net, optimizer, device):
    
    if buffer.length() < batch_size:
        return
    
    batch = buffer.sample(batch_size)
    state_batch = torch.stack(batch["states"]).squeeze()
    action_batch = np.array(batch["actions"])
    action_batch = torch.tensor(action_batch.reshape(action_batch.shape[0],1), dtype=torch.int64)
    reward_batch = torch.tensor(batch["rewards"])
    next_state_batch = batch["next_states"]
    done = torch.tensor(batch["status"])

    idx = [i for i in range(len(done)) if done[i]==False]
    non_final_next_states = torch.stack([next_state_batch[i] for i in idx]).squeeze()

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    #

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[idx] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    

    criterion = torch.nn.SmoothL1Loss()
    #print(state_action_values)
    #print(expected_state_action_values)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    #print(loss)
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train(target_net, policy_net, env, buffer, num_episodes =50, device= None):

    for i_episode in range(num_episodes):

        # Initialize the environment and get it's state
        state = env.reset()

        #flattening the state
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        while True:
            action = select_action(policy_net, state, env.action_list)
            observation, reward, done = env.perform_action(action)
            env.show()
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            buffer.append(state, action, next_state, reward,done)

            state = next_state

            optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

            optimize_model(buffer, policy_net, target_net, optimizer, device)

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                print("trained for episode: ",i_episode)
                break