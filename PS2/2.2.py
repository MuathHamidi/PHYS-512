#==================================================
# Course: PHYS 512
# Problem: PS2 P2
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
# Functions
#==================================================
def heaviside(x):
    return 1.0*(x>0)

def offset_gauss(x):
    return 1+10*np.exp(-0.5*x**2/(0.1)**2)

def cos(x):
    return np.cos(x)

#==================================================
# Integrate Adaptive
#==================================================
def integrate_adaptive(fun,a,b,tol,extra=None):
    global counter
    if extra==None:
        xs = np.linspace(a,b,5) # x partition
        dx = xs[1] - xs[0] 
        y = fun(xs)
        counter += len(y)
        area1 = (y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])*dx/3 # 3-point with dx
        area2 = (y[0]+4*y[2]+y[4])*(2*dx)/3 # 3-point with 2dx
        Error = np.abs(area1-area2)
        if Error < tol:
            return area1
        else:
            mid = (a+b)/2
            int1 = integrate_adaptive(fun, a, mid, tol/2, extra=[y[0], y[1], y[2], dx])
            int2 = integrate_adaptive(fun, mid, b, tol/2, extra=[y[2], y[3], y[4], dx])
            return int1 + int2
    
    else:   
        x = np.array([a+0.5*extra[3],b-0.5*extra[3]])
        y = fun(x)
        counter += len(y)
        dx = extra[3]/2
        area1 = dx*(extra[0]+4*y[0]+2*extra[1]+4*y[1]+extra[2])/3
        area2 = 2*dx*(extra[0]+4*extra[1]+extra[2])/3
        Error = np.abs(area1-area2)
        if Error < tol:
            return area1
        else:
            mid=(a+b)/2
            int1 = integrate_adaptive(fun, a, mid, tol/2, extra = [extra[0] ,y[0], extra[1], dx])
            int2 = integrate_adaptive(fun, mid, b, tol/2, extra = [extra[1] ,y[1], extra[2], dx])
            return int1 + int2  

#==================================================
# Integration (Jon's Function)
#==================================================
def Integration(fun,a,b,tol): # This is Jon's function with modification
    global counter
    xs = np.linspace(a,b,5) # x partition
    dx = xs[1] - xs[0] 
    y = fun(xs)
    counter += len(y)
    area1 = (y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])*dx/3 # 3-point with dx
    area2 = (y[0]+4*y[2]+y[4])*(2*dx)/3 # 3-point with 2dx
    Error = np.abs(area1-area2)
    if Error < tol:
        return area1
    else:
        mid = (a+b)/2
        int1 = Integration(fun, a, mid, tol/2)
        int2 = Integration(fun, mid, b, tol/2)
        return int1 + int2

#==================================================
# Calculations
#==================================================
# Heaviside
# Integration Adaptive
counter = 0
Integ = integrate_adaptive(heaviside, a=0.1, b=1, tol=1e-6, extra=None)
print("Number of function calls, Heaviside, Mine:", counter)
# Jon's Function
counter = 0
Integ = Integration(heaviside, a=0.1, b=1, tol=1e-6)
print("Number of function calls, Heaviside, Jon's:", counter)

# Cos
# Integration Adaptive
counter = 0
Integ = integrate_adaptive(cos, a=-1, b=1, tol=1e-6, extra=None)
print("Number of function calls, Cos, Mine:", counter)
# Jon's Function
counter = 0
Integ = Integration(cos, a=-1, b=1, tol=1e-6)
print("Number of function calls, Cos, Jon's:", counter)

# Gauss
# Integration Adaptive
counter = 0
Integ = integrate_adaptive(offset_gauss, a=-1, b=1, tol=1e-6, extra=None)
print("Number of function calls, Gauss, Mine:", counter)
# Jon's Function
counter = 0
Integ = Integration(offset_gauss, a=-1, b=1, tol=1e-6)
print("Number of function calls, Gauss, Jon's:", counter)
