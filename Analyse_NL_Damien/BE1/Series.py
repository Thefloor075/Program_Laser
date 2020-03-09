import numpy as np
import matplotlib.pyplot as plt

mu = 0
sigma = 1
size_signal_input = 1024

def Y(size_signal_input=1024, X = np.random.normal(mu, sigma, size_signal_input)):
	Z = np.ones(14)
	#Y = np.ones(size_signal_input)
	return 1/14 * np.convolve(X,Z)[7:size_signal_input+7]

def Func_DSP(Y):
	FFTF = np.fft.fft(Y)
	FFT = abs(FFTF)
	return FFT*FFT
	

#Génération d'un signal random
X_s = np.random.normal(mu, sigma, size_signal_input)

#Génération de la série y
Y_s = Y(size_signal_input, X_s)

DSP_X = Func_DSP(X_s)
DSP_Y = Func_DSP(Y_s)

#plt.plot(X_s)
#plt.plot(Y_s)

#plt.plot(DSP_X)
#plt.plot(DSP_Y)
plt.show()


#Question 3


yn1 = 0.1
Y = [0.1]
for _ in range(4095):
	yn1 = 1 - 2 * yn1*yn1
	Y.append(yn1)

Y = np.array(Y)
X = np.random.rand(4095)

S = 1/np.pi * np.arccos(-Y)

S_m = np.mean(S)
X_m = np.mean(X)
print("Moyenne S : {}, Y : {}".format(S_m, X_m))

S_v = np.var(S)
X_v = np.var(X)

print("Variance S : {}, Y : {}".format(S_v, X_v))

DSP_x = Func_DSP(X)
DSP_s = Func_DSP(S)

#plt.plot(DSP_x[1::])
#plt.plot(DSP_s[1::])
plt.show()




