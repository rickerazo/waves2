from __future__ import division
# Georgia State University
# Neuroscience Institue
# Ricardo Erazo
# traveling waves in HH endogenous oscillatory neurons.
# dynamic variable decaying synapse implementation. Parallel simulation

#the purpose of this script is to provide a brief inhibitory pulse to the neurons at hand
#and measure its response. This is used to figure out the most appropriate way to shock
#neurons to allow complete depolarization burst.
#A second purpose of this script is to note the difference in burst duration as a function of
#parameter K2theta

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

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
	aa = np.transpose(np.nonzero(v>Vt))
	s[aa] = s[aa]+delta2
	ab = np.transpose(np.nonzero(s>1))
	s[ab] =1
	ds = -s/tau2
	temp1 = np.concatenate((dv,dh),axis=0)
	temp2 = np.concatenate((dm,dn,ds),axis=0)
	f = np.concatenate((temp1,temp2),axis=0)
	#plt.plot(s)
	#plt.title(str(t))
	#print(t,'	',aa[0],'		',s)
	print(t,'	',aa,'		',ab,'		',v,'		',s)
	return f

def Vt_cross_ctr(t,y0): return y0[ctr]-Vt
#Vt_cross_ctr.terminal= True
Vt_cross_ctr.terminal= False
Vt_cross_ctr.direction = 1

nr_neurons = 3
delta1 = 1e-2
sigma1 = 2e-2
gsyn= 1
domain = np.arange(0,nr_neurons)*delta1
domain = np.array([domain])
Vt = -0.01
tau2 = 0.1
tspan = [0,13]
#tspan = [0,50]
delta2 = 0.1

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
#Htheta= 0.041325
#K2theta= -0.0060
#K2theta= -0.0065
#K2theta= -0.0070
#K2theta= -0.0075
#K2theta= -0.008
#K2theta= -0.0085
K2theta= -0.0088
#K2theta= -0.009 #duty cycle too active. 0.5 approx

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
v0= -0.0421996958
h0= 0.99223221
m0= 0.297443439
n0= 0.0152445526
## shocked
v2 = -4.4203449e-2
h2 = 9.92877246e-1
m2 = 3.02937855e-1
n2 = 1.52351209e-2


#initial conditions
y0 = np.array((v0,h0,m0,n2,1))
y2 = np.array((v2,h2,m2,n2))
initconds = np.zeros((1,nr_neurons*5))
initconds[0,0:nr_neurons] = v0
initconds[0,nr_neurons:nr_neurons*2] = h0
initconds[0,nr_neurons*2:nr_neurons*3] = m0
initconds[0,nr_neurons*3:nr_neurons*4] = n0
initconds[0,nr_neurons*4:nr_neurons*5] = 0
initconds[0,0] = v2
initconds[0,nr_neurons] = h2
initconds[0,nr_neurons*2]= m2
initconds[0,nr_neurons*3] = n2
initconds[0,nr_neurons*4] = 0
hmin = 1e-4
tev = np.arange(tspan[0],tspan[1],hmin)
spike_times = np.zeros((1,nr_neurons))
#for ctr in range(0,nr_neurons):
	#plt.figure(figsize=(10,10))
ctr = 1
neural_net = solve_ivp(evolve, tspan,initconds[0],method='RK45',t_eval = tev,events =Vt_cross_ctr ,atol=1e-7,rtol=1e-5)
	#time_spike= neural_net.t_events[0]
	#spike_times[0,ctr]=neural_net.t_events[0]
	#print('Wave'+str(ctr),spike_times[0,ctr])

fgsz = 9
plt.figure(figsize=(fgsz,fgsz))
for i in range(0,nr_neurons):
		plt.plot(neural_net.t,neural_net.y[i,:]-0.1*i)
plt.savefig('V_g'+str(gsyn)+'.png')
plt.figure(figsize=(fgsz,fgsz))
for i in range(0,nr_neurons):
		plt.plot(neural_net.t,neural_net.y[i+nr_neurons*4,:]-0.1*i)
plt.title('K2theta='+str(K2theta)+'_gsyn'+str(gsyn))
plt.savefig('S_g'+str(gsyn)+'.png')
np.save('s_g'+str(gsyn),neural_net.y[nr_neurons*4,:])
np.save('v_g'+str(gsyn),neural_net.y[0,:])
np.save('t_g'+str(gsyn),neural_net.t)
