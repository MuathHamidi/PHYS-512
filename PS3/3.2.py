# Course: PHYS 512
# Problem: PS3 P2
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
from scipy.integrate import solve_ivp # For math

#==================================================
# Times
#==================================================
minutes = 60 # minutes to seconds
hours = 60 * minutes # hours to seconds
days = 24 * hours # days to seconds
years = 365.25 * days # years to seconds

#==================================================
# Half Lives
#==================================================
U238 = 4.468e9 * years
Th234 = 24.1 * days
Pr234 = 6.7 * hours
U234 = 245500 * years
Th230 = 75380 * years
Ra226 = 1600 * years
Rn222 = 3.8235 * days
Po218 = 3.1 * minutes
Pb214 = 26.8 * minutes
Bi214 = 19.9 * minutes
Po214 = 164.3e-6
Pb210 = 22.3 * years
Bi210 = 5.015 * years
Po210 = 138.376 * days

half_life = [U238, Th234, Pr234, U234, Th230, Ra226, Rn222, Po218, Pb214, Bi214, Po214, Pb210, Bi210, Po210]

#==================================================
# Parameters
#==================================================
steps = len(half_life)

#==================================================
# Decay Solver
#==================================================
def Decay(x, y, half_life=half_life):
    dydx = np.zeros(steps+1) # active elements + Pb206
    
    dydx[0] = -y[0] / half_life[0] # for U238
    
    for i in range(1, steps):
        dydx[i] = y[i-1] / half_life[i-1] - y[i] / half_life[i]
    
    dydx[steps] = y[steps-1] / half_life[steps-1] # for Pb206
    
    return dydx

#==================================================
# Solve
#==================================================
t0 = 0 # initial time
tf = half_life[0] # final time (U238 half life)
y0 = np.zeros(steps+1)
y0[0] = 1 # initial value

ans = solve_ivp(Decay, [t0, tf], y0, method='Radau')

#print(ans)
np.savetxt("anst.txt", ans.t) # save times
np.savetxt("ansy.txt", ans.y) # save values

#==================================================
# Ratio of Pb206 to U238 (Numerical)
#==================================================
plt.plot(ans.t / years, ans.y[14,:] / ans.y[0,:], linewidth=5)
plt.title("Ratio of Pb206 to U238, Numerical")
plt.xlabel("Time (years)")
plt.ylabel("Pb206/U238")
plt.savefig('3.2.1.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()

#==================================================
# Ratio of Pb206 to U238 (Analytical)
#==================================================
N_U238 = np.exp(-ans.t/U238) # remained U238 share
N_Pb206 = 1 - N_U238 # remained Pb206 share
Ratio_Pb206_U238 = N_Pb206 / N_U238 # Pb206/U238 ratio

plt.plot(ans.t / years, Ratio_Pb206_U238, linewidth=5, label="Analytical")
plt.plot(ans.t / years, ans.y[14,:] / ans.y[0,:], linewidth=5, ls=":", label="Numerical")
plt.title("Ratio of Pb206 to U238, Analytical")
plt.xlabel("Time (years)")
plt.ylabel("Pb206/U238")
plt.legend()
plt.savefig('3.2.2.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()

#==================================================
# Ratio of Th230 to U234
#==================================================
tf = 10**6 * years # final time (of order 10 U234 half life)
ans = solve_ivp(Decay, [t0, tf], y0, method='Radau')

plt.plot(ans.t / years, ans.y[4,:] / ans.y[3,:], linewidth=5)
plt.title("Ratio of Th230 to U234")
plt.xlabel("Time (years)")
plt.ylabel("Th230/U234")
plt.savefig('3.2.3.pdf', format='pdf', dpi=1200)
plt.show()
plt.close()


