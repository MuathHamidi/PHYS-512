#==================================================
# Course: PHYS 512
# Problem: PS1 P1
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
# Numerical Differentiator
#==================================================
def NDiff(f,x,d): # f:function, x:variable, d:delta
    NDiff = (8 * (f(x + d) - f(x - d)) - f(x + 2*d) + f(x - 2*d)) / (12*d)
    return NDiff

#==================================================
# Optimal δ
#==================================================
E = 10**-16 # machine precision

# For exp(x)
d1 = E**(1/5)

# For exp(0.01x)
d2 = E**(1/5) * 100

#==================================================
# Functions & Errors
#==================================================
f = np.exp
x = 3
d = 10**np.linspace(-16, 0, 1000)

Error1 = np.abs(np.exp(x) - NDiff(f, x, d))
Error2 = np.abs(0.01*np.exp(0.01*x) - NDiff(f, 0.01*x, 0.01*d)/100) # This adjustment for NDiff to insure that delta is also multiplied by 0.01 inside the function

#==================================================
# Plot
#==================================================
# First Plot for exp(x)
plt.loglog(d, Error1)
plt.scatter(d1, np.abs(np.exp(x) - NDiff(f, x, d1)), c='red', s=80) # Optimal δ Error
plt.title("Error vs $\delta$, $f(x) = exp(x)$")
plt.xlabel("$\delta$")
plt.ylabel("Error")
plt.savefig('Error1 vs delta.png', format='png', dpi=1200)
plt.show()

# Second Plot for exp(0.01x)
plt.close()
plt.loglog(d, Error2)
plt.scatter(d2, np.abs(0.01*np.exp(0.01*x) - NDiff(f, 0.01*x, 0.01*d2)/100), c='red', s=80) # Optimal δ Error
plt.savefig('Error2 vs delta.png', format='png', dpi=1200)
plt.title("Error vs $\delta$, $f(x) = exp(0.01 x)$")
plt.xlabel("$\delta$")
plt.ylabel("Error")
plt.show()