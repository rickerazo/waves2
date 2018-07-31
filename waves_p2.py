from __future__ import division
# Georgia State University
# Neuroscience Institue
# Ricardo Erazo
# traveling waves in HH endogenous oscillatory neurons.
# alpha synapse implementation. Parallel simulation
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

nr_neurons = 50
delta1 = 1e-2
sigma1 = 1e-2
gsyn= 1.5
domain = np.arange(0,nr_neurons)*delta1
domain = np.array([domain])
Vt = 0.022
tau2 = 1
tspan = [0,100]

neuronSpace = np.arange(0,nr_neurons*delta1,delta1)

J = np.zeros((nr_neurons,nr_neurons))
for i in np.arange(0,nr_neurons,1):
	J[:,i] = abs(neuronSpace - neuronSpace[i])
J = delta1*np.exp(-J/sigma1)/2/sigma1
J = J-np.diag(np.diag(J))
for i in np.arange(0,nr_neurons,1):
	J[i,i:nr_neurons] = 0
W = gsyn*J
#Experimental parameters
Htheta= 0.04133
K2theta= -0.0075

#Cell parameters
gNa= 105
ENa= 0.045
gK2= 30
EK=-0.07
gH= 4
EH= -0.021
gl= 8
El= -0.046
Vrev= 0.01
#Vrev= -0.0425 - 0.07

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
v0= -0.04220452
h0= 0.99225122
m0= 0.29849783
n0= 0.01681514
## shocked
v2 = -0.036892644544022
h2 = 0.867431828696752
m2 = 0.016203233894593
n2 = 0.175309528389848

#First shocked neuron
def evolve(t, y0):
	u = y0
	du = np.reshape(u, (4,nr_neurons))
	v = du[0,:]
	h = du[1,:]
	m = du[2,:]
	n = du[3,:]
	s = n*0
	mNass=1./(1.+np.exp(-150.*(v+0.0305)))
	mK2ss=1./(1.+np.exp(-83.*(v+K2theta)))
	if v.any()>Vt:
		ts = t
	s[np.nonzero(v>Vt)] =1
	s[np.nonzero(v<Vt)] =0
	s = s*np.exp(-(t-ts)/tau2)
	#print(ts)
	dv = (-1/Cm) * (gNa*mNass*mNass*mNass*h*(v-ENa)+ gK2*n*n*(v-EK) + gH*m*m*(v-EH) + gl*(v-El) + 0.006 +np.dot(W,s)*(v-Vrev))
	dh = (1/(1+ np.exp(500*(v+0.0325))) - h)/tau_h
	dm = (1/(1+2*np.exp(180*(v+Htheta))+np.exp(500*(v+Htheta))) -m)/tau_m
	dn = (mK2ss - n)/tau_k
	#ds = (1/(1+np.exp(-5000*(v-0.02))) - s)/0.01
	temp1 = np.concatenate((dv,dh),axis=0)
	temp2 = np.concatenate((dm,dn),axis=0)
	f = np.concatenate((temp1,temp2),axis=0)
	return f

def Vt_cross_ctr(t,y0): return y0[ctr]-Vt
Vt_cross_ctr.terminal= True
#Vt_cross_ctr.terminal= False
Vt_cross_ctr.direction = 1
def Vt_cross(t,y0): return y0[0]-Vt
Vt_cross.terminal= False
#Vt_cross.terminal= True
Vt_cross.direction = 1

#initial conditions
y0 = np.array((v0,h0,m0,n2))
y2 = np.array((v2,h2,m2,n2))
initconds = np.zeros((1,nr_neurons*4))
initconds[0,0:nr_neurons] = v0
initconds[0,nr_neurons:nr_neurons*2] = h0
initconds[0,nr_neurons*2:nr_neurons*3] = m0
initconds[0,nr_neurons*3:nr_neurons*4] = n0
#initconds[0,nr_neurons*4:nr_neurons*5] = 0
initconds[0,0] = v2
initconds[0,nr_neurons] = h2
initconds[0,nr_neurons*2]= m2
initconds[0,nr_neurons*3] = n2
#initconds[0,nr_neurons*4] = 1
#neural_net = solve_ivp(evolve, tspan,initconds[0],events=Vt_cross)
#plt.figure()
#plt.ion()
ts= np.zeros((1,nr_neurons))
s = ts[0]
for ctr in range(0,nr_neurons):
	neural_net = solve_ivp(evolve, tspan,initconds[0],events=Vt_cross_ctr,atol=1e-7,rtol=1e-5)
	if np.size(neural_net.t_events[0])>0:
		spikes = neural_net.t_events[0]
		ts[0,ctr] = spikes[0]
	#print(ts[0,ctr],ctr)
	#plt.plot(neural_net.t,neural_net.y[ctr,:],label=str(ctr))
#plt.legend()

c = delta1/np.diff(ts[0])
c_par = c
a = np.diff(c)/np.diff(ts[0,0:-1])

plt.figure()
plt.ion()
plt.plot(delta1*np.arange(0,np.size(c),1),c)
plt.title('Speed, gsyn='+str(gsyn))
plt.ylabel('c')
plt.xlabel('distance')

#plt.figure()
#plt.ion()
#plt.plot(delta1*np.arange(0,np.size(a),1),a)

