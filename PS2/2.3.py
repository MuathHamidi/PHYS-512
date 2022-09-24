#==================================================
# Course: PHYS 512
# Problem: PS2 P3
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
import numpy.polynomial.chebyshev as chebyshev
from scipy.special import legendre

#==================================================
# Chebyshev
#==================================================
def Chebyshev(z):
    x = np.linspace(0.5,1,101)
    rescaled_x = 4 * x - 3
    y = np.log2(x)
    order = 7 # Using this degree gives 1e-07 to 1e-06 error magnitude. Which is what we want. Known by plotting later.
    return chebyshev.chebval(4 * z - 3, chebyshev.chebfit(rescaled_x, y, order))

#==================================================
# Plot Chebyshev
#==================================================
z = np.linspace(0.5,1,101)
rms = np.sqrt(np.sum((np.log2(z) - Chebyshev(z))**2)/len(z)) # RMS in Chebyshev
print("RMS in Chebyshev:",rms)
plt.plot(z, np.log2(z), label="$log_2$")
plt.plot(z, Chebyshev(z), ls=':', lw=5, label="Chebyshev")
plt.title("Chebyshev")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('2.3.1.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()

Error = Chebyshev(z) - np.log2(z)
plt.plot(z, Error) # Error plot
plt.title("Error in Chebyshev")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('2.3.2.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()

#==================================================
# mylog2
#==================================================
def mylog2(z):
    mantissa, exponent = np.frexp(z)
    x = np.linspace(0.5,1,101)
    rescaled_x = 4 * x - 3
    y = np.log(x)
    order = 7
    cheb = chebyshev.chebval(4 * mantissa - 3, chebyshev.chebfit(rescaled_x, y, order))
    return exponent * np.log(2) + cheb

#==================================================
# Plot mylog2
#==================================================
z = np.linspace(0.01,100,10000)
rms = np.sqrt(np.sum((np.log(z) - mylog2(z))**2)/len(z)) # RMS in mylog2
print("RMS in mylog2:",rms)
plt.plot(z, np.log(z), label="$log$")
plt.plot(z, mylog2(z), ls=':', lw=5, label="mylog2")
plt.title("mylog2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('2.3.3.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()

Error = mylog2(z) - np.log(z)
plt.plot(z, Error) # Error plot
plt.title("Error in mylog2")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('2.3.4.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()

#==================================================
# Legendre Polynomial (Bonus)
#==================================================
def Legendre(z):
    x = np.linspace(0.5,1,101)
    rescaled_x = 4 * x - 3
    order = 7
    y = legendre(order)(z)
    cheb = chebyshev.chebval(4 * z - 3, chebyshev.chebfit(rescaled_x, y, order))
    return cheb

#==================================================
# Plot Legendre
#==================================================
order = 7
z = np.linspace(0.5,1,101)
rms = np.sqrt(np.sum((legendre(order)(z) - Legendre(z))**2)/len(z)) # RMS in Legendre
print("RMS in Legendre:",rms)
plt.plot(z, legendre(order)(z), label="$Legendre true$")
plt.plot(z, Legendre(z), ls=':', lw=5, label="Legendre")
plt.title("Legendre")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig('2.3.5.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()

Error = Legendre(z) - legendre(order)(z)
plt.plot(z, Error) # Error plot
plt.title("Error in Legendre")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('2.3.6.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()