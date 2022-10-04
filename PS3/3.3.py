# Course: PHYS 512
# Problem: PS3 P3
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
# Data
#==================================================
data = np.loadtxt('dish_zenith.txt')
x = np.array(data[:,0])
y = np.array(data[:,1])
z = np.array(data[:,2])

#==================================================
# Matrix
#==================================================
A = np.zeros([len(x),4])
A[:,0] = x**2 + y**2 # with a
A[:,1] = x # with b
A[:,2] = y # with c
A[:,3] = 1 # with d

#==================================================
# Fit
#==================================================
lhs = A.T@A
rhs = A.T@z
v = np.linalg.inv(lhs)@rhs
z_pred = A@v

#==================================================
# Parameters
#==================================================
a, b, c, d = np.linalg.inv(lhs)@rhs

# Our parameters
x0 = b/(-2*a)
y0 = c/(-2*a)
z0 = d - a * (x0**2 + y0**2)
print("The best-fit parameters: a = {}, x0 = {}, y0 = {}, z0 = {}".format(a, x0, y0, z0))

#==================================================
# Error
#==================================================
Noise = np.mean((z - z_pred)**2)  # noise
uncertainty = np.sqrt(Noise * np.diag(np.linalg.inv(lhs)))
print("The uncertainty in a is {}".format(uncertainty[0]))

#==================================================
# Error Bar
#==================================================
focal_length = 1/(4*a) # focal length
print("Focal length =", focal_length)

Error = 1/4 * 1/(a)**2 * uncertainty[0]
print("Focal length error =", Error)