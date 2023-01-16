from random import sample

class buffer():
    def __init__(self) -> None:
         self.states = []
         self.actions = []
         self.next_states = []
         self.rewards = []
         self.status = []

    def append(self, state, action, state_new, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(state_new)
        self.rewards.append(reward)
        self.status.append(done)

    def sample(self,batch_size):
        idx = sample(range(self.length()),batch_size)
        return {"states":[self.states[i] for i in idx], "actions": [self.actions[i] for i in idx], "next_states": [self.next_states[i] for i in idx],\
            "rewards": [self.rewards[i] for i in idx], "status": [self.status[i] for i in idx]}

    def length(self):
        return len(self.states)