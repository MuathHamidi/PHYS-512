#==================================================
# Course: PHYS 512
# Problem: PS6 P1
#==================================================
# By: Muath Hamidi
# Email: muath.hamidi@mail.mcgill.ca
# Department of Physics, McGill University
# November 2022

#==================================================
# Libraries
#==================================================
import numpy as np # For math
import matplotlib.pyplot as plt # For graphs

#==================================================
# Shift Function
#==================================================
def Shift(Array, shift):
    DFT_Arr = np.fft.fft(Array)
    delta = np.zeros(len(Array))
    delta[shift] = 1 # This is to make sure we have the same amplitude
    DFT_del = np.fft.fft(delta)
    return np.fft.ifft(DFT_del * DFT_Arr)

#==================================================
# Plot
#==================================================
x = np.linspace(-5,5,100)
Gauss = np.exp(-0.5*x**2)

shift = 0
plt.plot(np.abs(Gauss), c="blue")
plt.axvline(len(Gauss)//2+shift, c="blue", ls = '--', label = "shift={}".format(shift))

shift = 50
plt.plot(np.abs(Shift(Gauss,shift)), c="orange")
plt.axvline(len(Gauss)//2+shift, c="orange", ls = '--', label = "shift={}".format(shift))

shift = 20
plt.plot(np.abs(Shift(Gauss,shift)), c="green")
plt.axvline(len(Gauss)//2+shift, c="green", ls = '--', label = "shift={}".format(shift))

plt.title("Convolution Shift Function - Gaussian")
plt.legend()
plt.show()
