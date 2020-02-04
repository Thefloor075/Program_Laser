import numpy as np	
import matplotlib.pyplot as plt
from pylab import show,subplot,figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

c1 = 5.56e-9
c2 = 50e-9
L = 7.14e-3
Bp = 1
m0 = - 0.5e-3
m1 = - 0.8e-3



#dt
dt = 1e-7
intervalle = 4e-3

n = int(intervalle/dt)


n_tr = int(0.25 * n)


def f(ur):
	return m0 * ur + 0.5 * (m1 - m0) * ( abs(ur + Bp) - abs(ur - Bp) )

def F(y,G_i):
	uc1 = y[0]
	uc2 = y[1]
	il = y[2]
	return np.array([ 1/c1 * ( G_i*(uc2 - uc1) - f(uc1) ), 1/c2 * ( G_i*(uc1 - uc2) + il ), -uc2/L])
	
def runge_kt4(y,G):
	k1 = dt*F(y,G)
	k2 = dt*F(y + k1*0.5,G)
	k3 = dt*F(y + k2*0.5,G)
	k4 = dt*F(y + k3,G)
	return y + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
	
def main():
	G = np.linspace(0.4, 0.8, 10)

	G = 1e-3*np.array(G)
	CI = np.array([1e-3,0,0])

	Y = []
	Temps = np.linspace(0,intervalle,int(intervalle/dt))
	for element in G:
		print("G : ",element)
		yn1 = runge_kt4(CI,element)
		Y.append(yn1)
		for i in range(1,n):
			yn1 = runge_kt4(yn1,element)
			Y.append(yn1)
		Y = np.array(Y)
		x = Y[n_tr:n,0]
		y = Y[n_tr:n,1]
		z = Y[n_tr:n,2]
		Temps = Temps[n_tr:n]
		subplot(221)
		plt.plot(Temps,x)
		plt.ylabel('x')
		plt.xlabel('Temps')
		subplot(222)
		plt.ylabel('y')
		plt.xlabel('Temps')
		plt.plot(Temps,y)     
		subplot(223)
		plt.ylabel('z')
		plt.xlabel('Temps')
		plt.plot(Temps,z)
		plt.savefig('XYZ_for_G_{}.png'.format(element))
		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot3D(x, y, z, 'gray')
		plt.savefig('Attracteur_for_G_{}.png'.format(element))
		Y = []




if __name__ == "__main__":
	main()
