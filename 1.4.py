#==================================================
# Course: PHYS 512
# Problem: PS1 P4
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

fun = np.cos
x = np.linspace(-np.pi, np.pi, 1001)

order = 9
x_points = np.linspace(-np.pi, np.pi, order)
m = 4
n = 4

#==================================================
# Polynomial Fit
#==================================================
# We have optimized functions for the Polynomial Fit in Python
def PolynomialFit(fun, x, order):
    poly = np.polyval(np.polyfit(x_points, fun(x_points), order), x)
    return poly

#==================================================
# Cubic Spline
#==================================================
def Spline(fun, x):
    cs = CubicSpline(x_points, fun(x_points))
    return cs(x)

#==================================================
# Rational
#==================================================
def Rational(fun, x, order, m, n):
    # This part is totally/partially from Jon with change of variables to suit my code
    pcols=[x_points**k for k in range(n+1)]
    pmat=np.vstack(pcols)

    qcols=[-x_points**k*fun(x_points) for k in range(1,m+1)]
    qmat=np.vstack(qcols)
    
    mat=np.hstack([pmat.T,qmat.T])
    coeffs=np.linalg.pinv(mat)@fun(x_points)
    
    num=np.polyval(np.flipud(coeffs[:n+1]),x)
    denom=1+x*np.polyval(np.flipud(coeffs[n+1:]),x)
    rational=num/denom
    
    return rational

#==================================================
# Call
#==================================================
poly = PolynomialFit(fun, x, order)
cs = Spline(fun, x)
rational = Rational(fun, x, order, m, n)

#==================================================
# Errors
#==================================================
PolyError = fun(x) - poly
SplineError = fun(x) - cs
RationalError = fun(x) - rational

#==================================================
# Plot
#==================================================
# Interpolations Plot
plt.plot(x, fun(x), label='cos(x)')
plt.scatter(x_points, fun(x_points), label='points')
plt.plot(x, poly, label='PolyFit')
plt.plot(x, cs, label="Spline")
plt.plot(x, rational, label="Rational")

plt.title("Cos(x) Interpolations, order={}".format(order))
plt.xlabel("x")
plt.ylabel("fun(x)")
plt.legend()
plt.show()

# Errors Plot
plt.close()
plt.plot(x, PolyError, label='PolyFit')
plt.plot(x, SplineError, label="Spline")
plt.plot(x, RationalError, label="Rational")

plt.title("Cos(x) Interpolations Errors, order={}".format(order))
plt.xlabel("x")
plt.ylabel("Error")
plt.legend()
plt.show()

#==================================================
#==================================================
#==================================================
# Lorantzian
#==================================================
x_points = np.linspace(-1,1,order)
x = np.linspace(-1,1,1001)

def fun(x):
    fun = 1/(1 + x**2)
    return fun

#==================================================
# Call
#==================================================
poly = PolynomialFit(fun, x, order)
cs = Spline(fun, x)
rational = Rational(fun, x, order, m, n)

#==================================================
# Errors
#==================================================
PolyError = fun(x) - poly
SplineError = fun(x) - cs
RationalError = fun(x) - rational

#==================================================
# Plot
#==================================================
# Interpolations Plot
plt.close()
plt.plot(x, fun(x), label='Lorentzian')
plt.scatter(x_points, fun(x_points), label='points')
plt.plot(x, poly, label='PolyFit')
plt.plot(x, cs, label="Spline")
plt.plot(x, rational, label="Rational")
plt.title("Lorentzian Interpolations, order={}".format(order))
plt.xlabel("x")
plt.ylabel("fun(x)")
plt.legend()
plt.show()

# Errors Plot
plt.close()
plt.plot(x, PolyError, label='PolyFit')
plt.plot(x, SplineError, label="Spline")
plt.plot(x, RationalError, label="Rational")

plt.title("Lorentzian Interpolations Errors, order={}".format(order))
plt.xlabel("x")
plt.ylabel("Error")
plt.legend()
plt.show()
