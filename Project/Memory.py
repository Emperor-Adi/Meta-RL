import numpy as np

class Memory:
    def __init__(self):
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_gae_r = []
        self.batch_s_ = []
        self.batch_done = []
        self.GAE_CALCULATED_Q = False


    def store(self, s, a, s_, r, done):
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)


    def get_batch(self,batch_size):
        s,a,gae_r = [],[],[]
        for pos in range(batch_size):
            pos = np.random.randint(len(self.batch_s))
            s.append(self.batch_s[pos])
            a.append(self.batch_a[pos])
            gae_r.append(self.batch_gae_r[pos])
        return (s, a, gae_r)


    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()
        self.GAE_CALCULATED_Q = False


    @property
    def sample_count(self):
        return len(self.batch_s)
