from scipy.linalg import solve
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

sigma = 8
rho = 28
beta = 8/3

a = 0.1
CI = np.array([a, a, a])

dt = 1e-3

def F(L):
	x = L[0]
	y = L[1]
	z = L[2]
	return np.array([ sigma * ( y - x ), rho * x - y - x * z, x * y - beta * z])
	
def runge_kt4(y):
	k1 = dt*F(y)
	k2 = dt*F(y + k1*0.5)
	k3 = dt*F(y + k2*0.5)
	k4 = dt*F(y + k3)
	return y + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

T = 100 

N = int(T/dt)
N1 = 10
Y = []
Y.append(CI)
#Initialisation



Yn1 = runge_kt4(CI)
Y.append(Yn1)
for i in range(1,N-1):
	Yn1 = runge_kt4(Yn1)
	Y.append(Yn1)


#Affichage De l'attracteur


Y = np.array(Y)

x = Y[:,0]
y = Y[:,1]
z = Y[:,2]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'gray')
plt.show()
"""
for N in range(0:20:200)
	x = Y[:N-2*N1,0]
	xT = Y[N1:N-N1,0]
	print(x.shape,xT.shape,x2T.shape)
	plt.scatter(x,xT, s=0.01)
	plt.show()
"""


