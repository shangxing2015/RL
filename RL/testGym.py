import random

from environment import _Noise

from util import  Counter

import numpy as np

init_state = np.arange(6)

x = init_state.reshape((2,3))

obs = np.zeros(3).reshape(1,3)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

c = np.concatenate((a,b),axis=0)

d = np.concatenate((x[1:][0:],obs),axis=0)

print d