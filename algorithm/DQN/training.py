import random
import math
import torch
from replay_buffer import memory


batch_size = 128
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
tau = 0.005
lr = 1e-4
steps_done = 0

#either select random action or action giving higher q-value
def select_action(policy_net, state, actions):

    global steps_done
    sample = random.random()

    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return random.sample(actions)



def optimize_model(memory, policy_net, target_net, optimizer, device):
    
    if memory.length() < batch_size:
        return
    
    batch = memory.sample(batch_size)

    state_batch = torch.tensor(batch["states"])
    action_batch = torch.tensor(batch["actions"])
    reward_batch = torch.tensor(batch["rewards"])
    next_state_batch = torch.tensor(batch["next_states"])
    done = torch.tensor(batch["status"])

    idx = [i for i in range(len(done)) if done[i]==False]
    non_final_next_states = torch.tensor([next_state_batch[i] for i in idx])

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[idx] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train(target_net, policy_net, device,env):
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):

        # Initialize the environment and get it's state
        state = env.reset()


        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        while True:
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break