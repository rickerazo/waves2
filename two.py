import numpy as np
from matplotlib import pyplot as plt

n0_t = np.load('t_g0.npy')
n0_v = np.load('v_g0.npy')
n0_s = np.load('s_g0.npy')

n1_t = np.load('t_g1.npy')
n1_v = np.load('v_g1.npy')
n1_s = np.load('s_g1.npy')


plt.figure(figsize=(10,10))
plt.ion()
plt.plot(n1_t,n1_v,label='1')
plt.plot(n0_t,n0_v,label='0')
#plt.plot(n1_t,n1_v,label='1')
plt.savefig('v.png')

plt.figure(figsize=(10,10))
plt.plot(n1_t,n1_s,label='1')
plt.plot(n0_t,n0_s,label='0')
#plt.plot(n1_t,n1_s,label='1')
plt.savefig('s.png')
