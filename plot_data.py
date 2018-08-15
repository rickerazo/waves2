import matplotlib
from matplotlib import pyplot as plt
import numpy as np
#current problem in analysis. saving c data. must save ts. raw data is more complete
ce = np.load('Vrev-20.npy')
ci = np.load('Vrev-62.npy')

font = {'family':'monospace',
		'weight':'bold',
		'size'	:20}
matplotlib.rc('font',**font)
delta1=1e-2

plt.figure(figsize=(10,10))
plt.ion()
plt.plot(delta1*np.arange(0,np.size(ce)),ce,label='Excitatory')
plt.plot(delta1*np.arange(0,np.size(ci)),ci,label='Inhibitory')
plt.legend()
plt.xlabel('Distance')
plt.ylabel('instant speed c')
plt.title('Wave evolution, g=10')

plt.savefig('ce_ci.png')
