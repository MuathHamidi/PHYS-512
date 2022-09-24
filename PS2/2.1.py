#==================================================
# Course: PHYS 512
# Problem: PS2 P1
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
from scipy.integrate import quad # For math

#==================================================
# Parameters
#==================================================
# Take the whole constant beside the integral --> 1
R = 1 # R
Z = np.linspace(0, 3*R, 401) # z values, from 0 to 3R
u1 = -1 # Integral's lower limit
u2 = 1 # Integral's upper limit

#==================================================
# Integrand
#==================================================
def Integrand(u):
    Integrand = (z - R*u) / (R**2 + z**2 - 2*R*z*u)**(1.5)
    return Integrand

#==================================================
# Integration
#==================================================
def Integration(Integrand,u1,u2,tol): # This is Jon's function with modification
    us = np.linspace(u1,u2,5) # u partition
    du = us[1] - us[0] 
    y = Integrand(us)
    area1 = (y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])*du/3 # 3-point with du
    area2 = (y[0]+4*y[2]+y[4])*(2*du)/3 # 3-point with 2du
    Error = np.abs(area1-area2)
    if Error < tol:
        return area1
    else:
        mid = (u1+u2)/2
        int1 = Integration(Integrand, u1, mid, tol/2)
        int2 = Integration(Integrand, mid, u2, tol/2)
        return int1 + int2

# My Integration
MyE = []
for i in range(0, len(Z)):
    z = Z[i]
    if z == 1: # Here where we have singularity
        MyE.append(np.nan)
    else:
        MyE.append(Integration(Integrand,u1,u2,tol=1e-6))

# Quad Integration
Quad = []
for i in range(0, len(Z)):
    z = Z[i]
    ans = quad(Integrand, -1, 1)
    Quad.append(ans[0])

#==================================================
# Plot
#==================================================
plt.plot(Z, Quad, c='green', label="Quad")
plt.plot(Z, MyE, ls=':', lw=5, label="My E")
plt.title("$E$ vs $z/R$")
plt.xlabel("$z/R$")
plt.ylabel("$E$")
plt.legend()
plt.show
plt.savefig('2.1.pdf', format='pdf', dpi=1200)