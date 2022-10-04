#==================================================
# Course: PHYS 512
# Problem: PS3 P1
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
# Derivative Function
#==================================================
def fun(x,y):
    global counter
    counter +=1
    
    dydx = y / (1 + x**2)
    return dydx

#==================================================
# RK4 (Step)
#==================================================
def rk4_step(fun, x, y, h):
    k1 = h * fun(x, y)
    k2 = h * fun(x + h/2, y + k1/2)
    k3 = h * fun(x + h/2, y + k2/2)
    k4 = h * fun(x + h, y + k3)
    dy = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y + dy

#==================================================
# Parameters & Definitions
#==================================================
steps = 201
x = np.linspace(-20, 20, steps)
h = np.median(np.diff(x))
y = np.zeros(steps)
y[0] = 1 # boundary condition

#==================================================
# Numerical Solution
#==================================================
counter = 0
for i in range(steps-1):
    y[i+1] = rk4_step(fun, x[i], y[i], h)

#==================================================
# Analytical Solution
#==================================================
c0 = 1 / np.exp(np.arctan(-20)) # true solution amplitude
y_true = c0 * np.exp(np.arctan(x)) # true solution

#==================================================
# Plot
#==================================================
plt.plot(x, y, linewidth=5, label="Numerical RK4")
plt.plot(x, y_true, linewidth=5, ls=":", label="Analytical")
plt.title("Given Function Integration, step")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('3.1.1.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()

#==================================================
# Error & Function Evaluations
#==================================================
print("In rk4_step, the difference mean is", np.mean(abs(y_true-y)), "with", counter,"function evaluations.")

#==================================================
# RK4 (Stepd)
#==================================================
def rk4_stepd(fun, x, y, h):
    # Step size h
    y1 = rk4_step(fun, x, y, h)
    
    # Step size h/2
    y2i = rk4_step(fun, x, y, h/2)
    y2 = rk4_step(fun, x + h/2, y2i, h/2)
    
    return y2 + (y2 - y1) / 15

#==================================================
# Parameters & Definitions
#==================================================
y = np.zeros(steps)
y[0] = 1 # boundary condition

#==================================================
# Numerical Solution
#==================================================
counter = 0
for i in range(steps-1):
    y[i+1] = rk4_stepd(fun, x[i], y[i], h)

#==================================================
# Plot
#==================================================
plt.plot(x, y, linewidth=5, label="Numerical RK4")
plt.plot(x, y_true, linewidth=5, ls=":", label="Analytical")
plt.title("Given Function Integration, stepd")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('3.1.2.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()

#==================================================
# Error & Function Evaluations
#==================================================
print("In rk4_stepd, the difference mean is", np.mean(abs(y_true-y)), "with", counter,"function evaluations.")