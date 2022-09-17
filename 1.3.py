#==================================================
# Course: PHYS 512
# Problem: PS1 P3
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
from scipy.interpolate import CubicSpline # Fot interpolation

#==================================================
# Function
#==================================================
data = open("lakeshore.txt", 'r')

v, t = [], [] # Voltage & Temperature in the Data
for line in data:
  values = [float(s) for s in line.split()]
  v.append(values[1])
  t.append(values[0])

# Sort the data
v, t = zip(*sorted(zip(v, t)))

# Cubic Spline
cs = CubicSpline(v, t)
minv = 0.090681 # Minimum Voltage in Data
maxv = 1.64429 # Maximum Voltage in Data
vs = np.linspace(minv, maxv, 1000) # The period

#==================================================
# Plot the Data Set and the Cubic Spline
#==================================================
plt.plot(v, t, 'o', label='data')
plt.plot(vs, cs(vs), label="Spline")

plt.title("Temperature vs Voltage")
plt.xlabel("Voltage")
plt.ylabel("Temperature")
plt.legend()
plt.show()

#==================================================
# Voltage
#==================================================
V = 0.5 #np.linspace(0.2, 1, 5) # Change this

if type(V) == float:
    V = np.linspace(V, 10**10, 1)

#==================================================
# Prototype
#==================================================
def lakeshore(V,data):
    TT = [] # Temperature Array
    Error = [] # Errors Array
    for j in range(0, len(V)):
        T = cs(V[j]) # Temperature
        TT.append(T)
        
        # Error = interpolated temperature - average temperature of the 2 nearest data points
        D = [i - V[j] for i in v] 
        index = min([i for i in D if i > 0])
        t_avg = (t[D.index(index)-1] + t[D.index(index)]) / 2
        error = abs(T - t_avg)
        Error.append(error)
        
    TT = [float(i) for i in TT] # Make its elements float
    return TT, Error

#==================================================
# Results
#==================================================
print(lakeshore(V,data))