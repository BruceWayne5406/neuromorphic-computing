
#simulation of a LIF type SNN using euler's formula

import matplotlib.pyplot as plt

t=0
dt=1/100

v= -10
v_0= -60
tau=2

vs=[]

while t<10:
    vs.append(v)

    dv= -(v - v_0) /tau

    v+=dv*dt
    t+=dt

plt.plot(vs)
plt.show()