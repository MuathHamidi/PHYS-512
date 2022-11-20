#==================================================
# Course: PHYS 512
# Problem: PS6 P2
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
# Part a
#==================================================
# Correlation Function
#==================================================
def Correlation(f, g):
    dft_f = np.fft.fft(f)
    dft_g = np.fft.fft(g)
    correlation = np.fft.ifft(dft_f * np.conj(dft_g))
    return correlation

#==================================================
# Plot [Correlation Function - Gaussian with itself]
#==================================================
x = np.linspace(-5,5,100)
Gauss = np.exp(-0.5*x**2)

plt.plot(np.abs(Correlation(Gauss,Gauss)), label="Gaussian Correlation")
plt.title("Correlation Function - Gaussian with itself")
plt.legend()
plt.show()

#==================================================
# Part b
#==================================================
# Shifted Correlation Function
#==================================================
def Shifted_Correlation(Array, shift):
    return np.fft.ifft(np.fft.fft(Array) * np.fft.fft(Shift(Array, shift)))

#==================================================
# Plot [Shifted Correlation Function - Gaussian]
#==================================================
for i in range(5):
    shift = 20 * i
    plt.plot(np.abs(Shifted_Correlation(Gauss, shift)), label="shift={}".format(shift))
plt.title("Shifted Correlation Function - Gaussian")
plt.legend()
plt.show()
