#==================================================
# Course: PHYS 512
# Problem: PS6 P3
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
# Wrapless Convolution Function
#==================================================
def Wrapless_Convolution(f, g):
    f_pad = np.pad(f, (0, len(g) - 1)) # f padding
    g_pad  = np.pad(g, (0, len(f) - 1)) # g padding
    return np.fft.irfft(np.fft.rfft(f_pad) * np.fft.rfft(g_pad))

#==================================================
# Plot [Wrapless Convolution Function - Gaussian with itself]
#==================================================
x = np.linspace(-5,5,100)
Gauss = np.exp(-0.5*x**2)

plt.plot(Wrapless_Convolution(Gauss,Gauss))
plt.title("Wrapless Convolution Function - Gaussian with itself")
plt.show()
