from __future__ import division
# Georgia State University
# Neuroscience Institue
# Ricardo Erazo
# traveling waves in HH endogenous oscillatory neurons.
# dynamic variable decaying synapse implementation. Parallel simulation

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

#neurons shocked with inhibitory pulse. Primed to fire
nr_shocked = 1
#total number of neurons in network. Shocked included
nr_neurons = 4
delta1 = 1e-2
sigma1 = 2e-2
gsyn= 5
domain = np.arange(0,nr_neurons)*delta1
domain = np.array([domain])
Vt = -0.01
tau2 = 0.11
tspan = [0,20]
#tspan = [0,80]
delta2 = 0.05
#tau2 = 0.1
#delta2 = 0.1

neuronSpace = np.arange(0,nr_neurons*delta1,delta1)

J = np.zeros((nr_neurons,nr_neurons))
for i in np.arange(0,nr_neurons,1):
	J[:,i] = abs(neuronSpace - neuronSpace[i])
J = delta1*np.exp(-J/sigma1)/2/sigma1
J = J-np.diag(np.diag(J))
for i in np.arange(0,nr_neurons,1):
	J[i,i:nr_neurons] = 0
W = gsyn*J
W1 = W[0,:]
#Experimental parameters
Htheta= 0.04134
K2theta= -0.007

#Cell parameters
gNa= 105
ENa= 0.045
gK2= 30
EK=-0.07
gH= 4
EH= -0.021
gl= 8
El= -0.046
#Vrev= 0.01
#Vrev=-0.02
Vrev= -0.062

#time scales
Cm= 0.5
tau_h= 0.04050
tau_m= 0.1
tau_k= 2
tau_s= 0.1
tau1= Cm;
int_time = 50

# ODE
#initial conditions
## rest
v0= -0.0421582656
h0= 0.992059993
m0= 0.295518678
n0= 0.0159591202
## shocked
v2 = -0.0442034035
h2 = 0.992878381
m2 = 0.030293481
n2 = 0.0152342858

#First shocked neuron
def evolve(t, y0):
	u = y0
	du = np.reshape(u, (5,nr_neurons))
	v = du[0,:]
	h = du[1,:]
	m = du[2,:]
	n = du[3,:]
	s = du[4,:]
	mNass=1./(1.+np.exp(-150.*(v+0.0305)))
	mK2ss=1./(1.+np.exp(-83.*(v+K2theta)))
	dv = (-1/Cm) * (gNa*mNass*mNass*mNass*h*(v-ENa)+ gK2*n*n*(v-EK) + gH*m*m*(v-EH) + gl*(v-El) + 0.006 +np.dot(W,s)*(v-Vrev))
	dh = (1/(1+ np.exp(500*(v+0.0325))) - h)/tau_h
	dm = (1/(1+2*np.exp(180*(v+Htheta))+np.exp(500*(v+Htheta))) -m)/tau_m
	dn = (mK2ss - n)/tau_k
	aa = np.nonzero(v>Vt)
	s[aa] = s[aa]+delta2
	ab = np.nonzero(s>1)
	s[ab] =1
	ds = -s/tau2
	temp1 = np.concatenate((dv,dh),axis=0)
	temp2 = np.concatenate((dm,dn,ds),axis=0)
	f = np.concatenate((temp1,temp2),axis=0)
	return f

def Vt_cross_ctr(t,y0): return y0[ctr]-Vt
Vt_cross_ctr.terminal= True
#Vt_cross_ctr.terminal= False
Vt_cross_ctr.direction = 1

#initial conditions
y0 = np.array((v0,h0,m0,n2))
y2 = np.array((v2,h2,m2,n2))
initconds = np.zeros((1,nr_neurons*5))
initconds[0,0:nr_neurons] = v0
initconds[0,nr_neurons:nr_neurons*2] = h0
initconds[0,nr_neurons*2:nr_neurons*3] = m0
initconds[0,nr_neurons*3:nr_neurons*4] = n0
initconds[0,nr_neurons*4:nr_neurons*5] = 0
initconds[0,0:nr_shocked] = v2
initconds[0,nr_neurons:nr_neurons+nr_shocked] = h2
initconds[0,nr_neurons*2:nr_neurons*2+nr_shocked]= m2
initconds[0,nr_neurons*3:nr_neurons*3+nr_shocked] = n2
initconds[0,nr_neurons*4:nr_neurons*4+nr_shocked] = 0

f= open('sim_out.dat','w')
data_out = 'Vrev	gsyn	K2theta	Propagation'
f.write(data_out)
f.write('\n')
ts= np.zeros((1,nr_neurons))
propagation = True
ctr = 0
while ctr < nr_neurons:
#for ctr in range(0,nr_neurons):
	neural_net = solve_ivp(evolve, tspan,initconds[0],method='RK45',events=Vt_cross_ctr,atol=1e-7,rtol=1e-5)
	if np.size(neural_net.t_events[0])>0:
		spikes = neural_net.t_events[0]
		ts[0,ctr] = spikes[0]
		print(ts[0,ctr])
	else:
		propagation = False
		ctr = nr_neurons
	ctr = ctr+1
## plots 
fgsz = 9
#1 activation variable synaptic current
plt.figure(figsize=(fgsz,fgsz))
plt.ion()
for i in range(0,nr_neurons):
		plt.plot(neural_net.t,neural_net.y[i+nr_neurons*4,:]-0.5*i,label=str(i))
plt.title('Activation variable of synaptic current. tau'+str(tau2)+'delta'+str(delta2))
plt.ylabel('s')
plt.xlabel('time (s)')
plt.legend()
plt.savefig('s-tau'+str(tau2)+'delta'+str(delta2)+'K2theta'+str(K2theta)+'.png')

#2 v traces
plt.figure(figsize=(fgsz,fgsz))
plt.ion()
for i in range(0,nr_neurons):
		plt.plot(neural_net.t,neural_net.y[i,:]-0.05*i,label=str(i))
plt.title('Voltage traces. tau'+str(tau2)+'delta'+str(delta2))
plt.ylabel('Membrane potential (mV)')
plt.xlabel('time (s)')
plt.legend()
plt.savefig('v-tau'+str(tau2)+'delta'+str(delta2)+'K2theta'+str(K2theta)+'.png')

spike_times = ts[0,nr_shocked:nr_neurons]
c = delta1/np.diff(spike_times)
a = np.diff(c)/np.diff(spike_times[0:-1])

#c_fast = np.mean(c[np.nonzero(np.abs(a)<1e-3)])
#3 wave instant speed
plt.figure(figsize=(fgsz,fgsz))
plt.ion()
plt.plot(delta1*np.arange(0,np.size(c),1),c)
plt.title('Instant Speed. tau'+str(tau2)+'delta'+str(delta2))
plt.ylabel('c')
plt.xlabel('distance')
plt.legend()
plt.savefig('c-tau'+str(tau2)+'delta'+str(delta2)+'K2theta'+str(K2theta)+'.png')
#plt.savefig('g'+str(gsyn))

#plt.show()
np.save('ts-Vrev'+str(Vrev)+'g'+str(gsyn)+'K2theta'+str(K2theta),spike_times)
data_out = str(Vrev)+'	'+str(gsyn)+'		'+str(K2theta)+'	'+str(propagation)
f.write(data_out)
f.write('\n')
f.close()
