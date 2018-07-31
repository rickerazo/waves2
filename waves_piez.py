from __future__ import division
# Georgia State University
# Neuroscience Institue
# Ricardo Erazo
# traveling waves in HH endogenous oscillatory neurons.
# time dependent decaying synapse. Piece-wise simulation
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

nr_neurons = 350
delta1 = 1e-2
sigma1 = 1e-2
gsyn= 2
domain = np.arange(0,nr_neurons)*delta1
domain = np.array([domain])
Vt = 0.022
tau2 = 1
Ie = 1
Ispike = 0.3*Ie*np.exp(-np.arange(0,nr_neurons,1)*delta1/sigma1)
Isyn = Ie*np.exp(-np.arange(0,nr_neurons,1)*delta1/sigma1)

#Experimental parameters
Htheta= 0.041326
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
def vecf(t, y0):
	v,h,m,n = y0
	mNass=1./(1.+np.exp(-150.*(y0[0]+0.0305)))
	mK2ss=1./(1.+np.exp(-83.*(y0[0]+K2theta)))
	if t>t_last:
		Ie_time_dependent = Ie_local*np.exp(-(t-t_last)/tau2)
	else:
		Ie_time_dependent = 0
	#plt.plot(t,Ie_time_dependent,marker='*')
	dv = (-1/Cm) * (gNa*mNass*mNass*mNass*y0[1]*(y0[0]-ENa)+ gK2*y0[3]*y0[3]*(y0[0]-EK) + gH*y0[2]*y0[2]*(y0[0]-EH) + gl*(y0[0]-El) + 0.006 +gsyn*Ie_time_dependent*(y0[0]-Vrev))
	dh = (1/(1+ np.exp(500*(y0[0]+0.0325))) - y0[1])/tau_h
	dm = (1/(1+2*np.exp(180*(y0[0]+Htheta))+np.exp(500*(y0[0]+Htheta))) -y0[2])/tau_m
	dn = (mK2ss - y0[3])/tau_k
	f = dv, dh, dm, dn
	return f

def Vt_cross(t,y0): return y0[0]-Vt
Vt_cross.terminal= False
#Vt_cross.terminal= True
Vt_cross.direction = 1

#initial conditions
y0 = np.array((v0,h0,m0,n2))
y2 = np.array((v2,h2,m2,n2))
statevar = np.zeros((nr_neurons,4))
for i in np.arange(0,nr_neurons):
	statevar[i,:] = y0
t1= 0
ts = np.zeros((nr_neurons,1))
t_last = 0
Ie_local = Isyn[0]
Ie_local = 0
#plt.figure()
#plt.ion()
tspan_pre = [ts[0], ts[0]+int_time]
presynaptic_neuron = solve_ivp(vecf, tspan_pre, y2,events= Vt_cross, method = 'RK45',rtol=1e-5,atol=1e-7)
#presynaptic_neuron = solve_ivp(vecf, tspan_pre, y0,events= Vt_cross, method = 'RK45',rtol=1e-5,atol=1e-7)
if np.size(presynaptic_neuron.t_events)>0:
	tspk = presynaptic_neuron.t_events[0]
	ts[0] = tspk[0]
	tspan_post = [t1, ts[0]]
	for i in np.arange(1,nr_neurons):
		Ie_local = Isyn[i]
		postsynaptic_neuron = solve_ivp(vecf, tspan_post, statevar[i,:], method = 'RK45',rtol=1e-5,atol=1e-7)
		statevar[i,:] = postsynaptic_neuron.y[:,-1]
		#plt.plot(postsynaptic_neuron.t,postsynaptic_neuron.y[0,:])
#plt.plot(presynaptic_neuron.t,presynaptic_neuron.y[0,:])
t_last = ts[0]
ctr = 1
Isyn1 = Isyn
Isyn = Isyn*np.exp(-(ts[0] - t1)/tau2)
Isyn[ctr:nr_neurons] = Isyn[ctr:nr_neurons] + Ispike[0:nr_neurons-ctr]
Isyn[0:ctr] = 0
#plt.figure()
#plt.plot(Isyn1)
#plt.plot(Isyn)
for ctr in np.arange(1,nr_neurons):
	Ie_local = Isyn[ctr]
	tspan_neuron = [ts[ctr-1], ts[ctr-1]+int_time]
	neuron = solve_ivp(vecf, tspan_neuron, statevar[ctr,:],events= Vt_cross, method = 'RK45',rtol=1e-5,atol=1e-7)
	if np.size(neuron.t_events)>0:
		tspk = neuron.t_events[0]
		ts[ctr] = tspk[0]
		tspan_post = [ts[ctr-1], ts[ctr]]
		for i in np.arange(ctr+1,nr_neurons):
			Ie_local = Isyn[i]
			post_neuron = solve_ivp(vecf, tspan_post, statevar[i,:], method = 'RK45',rtol=1e-5,atol=1e-7)
			statevar[i,:] = post_neuron.y[:,-1]
			#plt.plot(post_neuron.t, post_neuron.y[0,:])
		t_last = tspk[0]
		Isyn = Isyn*np.exp(-(ts[ctr] - ts[ctr-1])/tau2)
		Isyn[ctr:nr_neurons] = Isyn[ctr:nr_neurons] + Ispike[0:nr_neurons-ctr]
		Isyn[0:ctr] = 0
	else: break
	#plt.plot(neuron.t,neuron.y[0,:],label=str(ctr))
speed = delta1/np.diff(np.transpose(ts))
c = speed[0]
c_piez = c
acceleration = np.diff(speed)/np.diff(np.transpose(ts[0:-1]))
a = acceleration[0,:]
plt.figure()
plt.ion()
plt.plot(delta1*np.arange(0,np.size(c)),c)
plt.title('Speed, gsyn='+str(gsyn))
plt.ylabel('c')
plt.xlabel('distance')
#plt.figure()
#plt.ion()
#plt.plot(delta1*np.arange(0,np.size(a)),a)
	#plt.ion()
	#plt.plot(c)


	#label1 = 'gsyn='
	#label2 = str(gsyn)
	#label3 = ',Vrev='
	#label4 = str(Vrev)
	#fname1 = np.core.defchararray.add(label1,label2)
	#fname2 = np.core.defchararray.add(label3,label4)
	#fname = str(np.core.defchararray.add(fname1,fname2))
	#fname = str(np.core.defchararray.add(fname,'.npy'))
	#np.save(fname,[c,a])

	#fh.close()
#else: print('wave failed')
