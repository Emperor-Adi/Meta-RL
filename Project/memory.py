import numpy as np

class Memory:
    def __init__(self) -> None:
        self.batch_s = [] # states
        self.batch_a = [] # actions
        self.batch_s_ = [] # next states
        self.batch_r = [] # rewards
        self.batch_gae_r = [] # GAE rewards
        self.batch_done = []
        self.GAE_CALCULATED_Q = False

    def store(self, s, a, s_, r ,done):
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_s_.append(s_)
        self.batch_r.append(r)
        self.batch_done.append(done)
    
    def get_batch(self, batch_size):
        s,a,s_,r,gae_r,d = [],[],[],[],[],[]
        rng = np.random.random(batch_size)
        rng = list(map(lambda x: int(self.size*x),rng))
        s.extend([self.batch_s[pos] for pos in rng])
        a.extend([self.batch_a[pos] for pos in rng])
        s_.extend([self.batch_s_[pos] for pos in rng])
        r.extend([self.batch_r[pos] for pos in rng])
        gae_r.extend([self.batch_gae_r[pos] for pos in rng])
        d.extend([self.batch_done[pos] for pos in rng])
        return (s,a,s_,r,gae_r,d)

    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()
        self.GAE_CALCULATED_Q = False
    
    @property
    def size(self):
        return len(self.batch_s)
