#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:04:27 2024

@author: Luki
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import json

# Load system parameters
with open('C://Users/Lukas/Documents/GitHub/PFC42/resultsC17/parameters/system-parameter-C15-230125.json', 'r') as file:
    system_params = json.load(file)

# Load final data stack
with open('C://Users/Lukas/Documents/GitHub/PFC42/Final-Results/finaldatastack.json', 'r') as file:
    final_data = json.load(file)

# Constants
e = 1.60217662e-19  # Elementary charge
eps0 = 8.854187817e-12  # Vacuum permittivity
u = 1.660539066 *10**(-27)
m_neon = 20.1797 * u #*u = kg
md = 6.64e-13  # Dust particle mass (assuming 1 μm radius melamine-formaldehyde)
k_b = 1.38065 * 10**(-23)   #m^2kg/s^2K

# Parameters for 15 Pa from the provided data
lambda_De = system_params['15pa']["debye_De"]
#lambda_De = np.sqrt((system_params['15pa']["T_e"])/(4*np.pi*(e**2)*system_params['15pa']["n_e"]))
w_pi = system_params['15pa']["w_pi"]  # Ion plasma frequency
lambda_D = system_params['15pa']["debye_D"]  # Debye length
Ti = system_params['15pa']["T_i"] * e  # Ion temperature in Joules
lambda_i = system_params['15pa']["debye_Di"] # Ion mean free path
nd = system_params['15pa']["n_d"]  # Dust density
ni = system_params['15pa']["n_i"]  # Ion density
E = abs(system_params['15pa']["e-field-vm"])  # Electric field strength
vd = final_data['15pa']["v_group_100"]

# Derived parameters
vTi = np.sqrt(k_b*system_params['15pa']["T_i"]*11600/(m_neon)) #(k_b*T_i/m_d)**(1/2) #particle thermal temperature # Ion thermal velocity
nu_in = vTi / system_params['15pa']["debye_Di"]  # Ion-neutral collision frequency
w_pd = system_params['15pa']["w_pd"]  # Dust plasma frequency
mu_i = e / (m_neon * nu_in)  # Ion mobility
ui = mu_i * E  # Ion drift velocity

# Dispersion relation function
def dispersion_relation(w, k):
    chi_e = 1 / (k * lambda_De)**2
    chi_i = w_pi**2 / (k**2 * vTi**2 + k * ui *(1j * nu_in - k * ui))
    chi_d = -w_pd**2 / (w + k*vd) * (w + k*vd + 1j*nu_in)  # Assuming dust-neutral collision frequency is 0.1 * nu_in
    return 1 + chi_e + chi_i + chi_d

# Solve dispersion relation
k_values = np.linspace(0, 50, 500)
w_real = np.zeros_like(k_values)
w_imag = np.zeros_like(k_values)

def equation(w, k):
    return [np.real(dispersion_relation(w[0] + 1j*w[1], k)), 
            np.imag(dispersion_relation(w[0] + 1j*w[1], k))]

for i, k in enumerate(k_values):
    sol = root(equation, [0, 0], args=(k,))
    w_real[i], w_imag[i] = sol.x

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(k_values, w_real / w_pi, label='Real part')
plt.plot(k_values, w_imag / w_pi, label='Imaginary part')
plt.xlabel('Wave number k (m^-1)')
plt.ylabel('ω / ω_pi')
plt.title('Dispersion Relation for Dust Acoustic Waves at 15 Pa')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print phase velocity at k = 1000 m^-1
k_example = 1000
sol_example = root(equation, [w_pi, 0], args=(k_example,))
w_example = sol_example.x[0] + 1j*sol_example.x[1]
v_phase = np.real(w_example) / k_example
print(f"Phase velocity at k = 1000 m^-1: {v_phase:.2f} m/s")

# Calculate group velocity
dk = 1e-6 * k_example
w_plus = root(equation, [w_pi, 0], args=(k_example + dk,)).x
w_minus = root(equation, [w_pi, 0], args=(k_example - dk,)).x
v_group = (w_plus[0] - w_minus[0]) / (2 * dk)
print(f"Group velocity at k = 1000 m^-1: {v_group:.2f} m/s")

# Compare with experimental data
k_exp = 1027.9320900440828  # From finaldatastack.json for 15 Pa
v_group_exp = 90.6602592365529  # From finaldatastack.json for 15 Pa
print(f"Experimental group velocity at k = {k_exp:.2f} m^-1: {v_group_exp:.2f} m/s")