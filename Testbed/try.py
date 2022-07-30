import numpy as np
l = [1,0,0,1,1,1]
l = np.ndarray(l)
ll = np.zeros(shape=(6,1))

ll[:,l.flatten()]=1
print(ll)