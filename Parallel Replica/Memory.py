import numpy as np

class Memory:
    def __init__(self) -> None:
        self.batch_state = []
        self.batch_action = []
        self.batch_reward = []
        self.batch_gae_reward = []
        self.batch_next_state = []
        self.batch_done = []
        self.GAE_CALCULATED_Q = False

    def store(self, state, action, reward, next_state, done):
        self.batch_state.append(state)
        self.batch_action.append(action)
        self.batch_reward.append(reward)
        self.batch_next_state.append(next_state)
        self.batch_done.append(done)
    
    def get_batch(self, batch_size):
        s,a,r,gae_r,s_,d = [],[],[],[],[],[]
        for _ in range(batch_size):
            pos = np.random.randint(len(self.batch_state))
            s.append(self.batch_state[pos])
            a.append(self.batch_action[pos])
            r.append(self.batch_reward[pos])
            gae_r.append(self.batch_gae_reward[pos])
            s_.append(self.batch_next_state[pos])
            d.append(self.batch_done[pos])
        return (s,a,r,gae_r,s_,d)
    
    def clear(self):
        self.batch_state.clear()
        self.batch_action.clear()
        self.batch_reward.clear()
        self.batch_next_state.clear()
        self.batch_done.clear()
        self.GAE_CALCULATED_Q = False
    
    @property
    def sample_count(self):
        return len(self.batch_state)