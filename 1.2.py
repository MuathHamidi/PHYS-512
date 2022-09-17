#==================================================
# Course: PHYS 512
# Problem: PS1 P2
#==================================================
# By: Muath Hamidi
# Email: muath.hamidi@mail.mcgill.ca
# Department of Physics, McGill University
# September 2022

#==================================================
# Libraries
#==================================================
import numpy as np # For math
import matplotlib.pyplot as plt # For graphs

#==================================================
# Function, Derivative, 3rd Derivative & Variable
#==================================================
fun = np.exp
funD = np.exp
funD3 = np.exp
x = np.linspace(-1, 1, 10)

#==================================================
# Numerical Differentiator
#==================================================
def ndiff(fun,x,full):
    dx = (10**-16 * fun(x) / funD3(x))**(1/3) # Optimal dx
    F = (fun(x + dx) - fun(x - dx)) / (2 * dx) # Differentiator
    error = np.abs(F - funD(x)) # Error
    if full == False:
        return F
    if full == True:
        return F, dx, error

#==================================================
# Numerical Differentiator Prototype
#==================================================
NDiff = ndiff(fun,x,full=True)

#==================================================
# Saving Results
#==================================================
result = open("result.out","w")
np.savetxt(result, np.c_[NDiff] )
result.close()