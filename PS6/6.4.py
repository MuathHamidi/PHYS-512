#==================================================
# Course: PHYS 512
# Problem: PS6 P4
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
# Part c
#==================================================
# Parameters
#==================================================
N = 100
k = np.arange(N) # k
x = np.arange(N) # x

#==================================================
# Functions
#==================================================
def F(kp):
    F = np.imag((1-np.exp(-2*1j*np.pi*(k-kp)))/(1-np.exp(-2*1j*np.pi*(k-kp)/N)))
    return F

def Sin(kp):
    sin = np.exp(2*np.pi*1j*kp*x/N)
    DFT = np.imag(np.fft.fft(sin))
    return DFT

#==================================================
# Plot
#==================================================
kp = 48 # k' integer
plt.plot(k, F(kp), label="Analytical Solution")
plt.scatter(k, Sin(kp), label="FFT", marker=".")
plt.legend()
plt.show()

kp = 48.3 # k' non-integer
plt.plot(k, F(kp), label="Analytical Solution")
plt.scatter(k, Sin(kp), label="FFT", marker=".")
plt.legend()
plt.show()

#==================================================
# Part d
#==================================================
# Functions
#==================================================
def Window(x):
    window = 0.5 - 0.5 * np.cos(2*np.pi*x/N)
    window = window / np.mean(window)
    return window

def NewSin():
    sin = np.exp(2*1j*np.pi*kp*x/N)
    new_sin = Window(x)*sin
    DFT = np.imag(np.fft.fft(new_sin))
    return DFT

#==================================================
# Plot
#==================================================
plt.plot(k, NewSin(), label="Windowed")
plt.plot(k, Sin(kp), label="Unwindowed")
plt.legend()
plt.show()

#==================================================
# Part e
#==================================================
# Plot
#==================================================
window = 0.5 - 0.5 * np.cos(2*np.pi*x/N)
plt.scatter(x, np.real(np.fft.fft(window)))

print(np.real(np.fft.fft(window)[:2]), np.real(np.fft.fft(window)[-1]))
