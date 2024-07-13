#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:54:52 2024

@author: Luki
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, e, k
from scipy.optimize import fsolve
import json

# Constants
electron_mass = 9.10938356e-31  # electron mass in kg
k_B = k  # Boltzmann constant in J/K

# Load system parameters
with open('/Users/Luki/Documents/GitHub/PFC42/resultsC17/parameters/system-parameter-C15-230125.json', 'r') as file:
    system_params = json.load(file)

# Given parameters for the plasma
params_20pa = system_params["20pa"]
params_25pa = system_params["25pa"]
params_30pa = system_params["30pa"]

# Wave number range
k_values = np.linspace(0.01, 10000, 5000)  # in 1/m, adjusted for the typical range of kλ_D

def solve_dispersion_relation_simplified(params):
    omega_r = []
    omega_i = []
    for k_val in k_values:
        # Solve the simplified dispersion relation for omega
        def dispersion_relation(omega):
            return omega**2 + 1j * params["beta_damp"] * omega - (params["w_pd"] * k_val**2 * params["debye_D"]) / (1 + k_val**2 * params["debye_D"]**2)
        
        # Initial guess for omega
        omega_initial_guess = np.sqrt((params["w_pd"] * k_val**2 * params["debye_D"]) / (1 + k_val**2 * params["debye_D"]**2))
        
        # Solve for the real and imaginary parts of omega
        omega_sol = fsolve(lambda omega: np.real(dispersion_relation(omega + 1j * params["beta_damp"])), omega_initial_guess)
        omega_imag_sol = np.imag(dispersion_relation(omega_sol[0]))
        
        # Combine the real and imaginary parts
        omega_r.append(omega_sol[0])
        omega_i.append(omega_imag_sol)
        
    return np.array(omega_r), np.array(omega_i)

# Solving the dispersion relation for each pressure
omega_r_20pa, omega_i_20pa = solve_dispersion_relation_simplified(params_20pa)
omega_r_25pa, omega_i_25pa = solve_dispersion_relation_simplified(params_25pa)
omega_r_30pa, omega_i_30pa = solve_dispersion_relation_simplified(params_30pa)

# Plotting the theoretical dispersion relation
plt.figure(figsize=(8, 5))
plt.plot(k_values* params_20pa["debye_D"], omega_r_20pa/params_20pa["w_pd"] , label='Real part (20 Pa)')
plt.plot(k_values* params_25pa["debye_D"], omega_r_25pa/params_25pa["w_pd"], label='Real part (25 Pa)')
plt.plot(k_values* params_30pa["debye_D"], omega_r_30pa/params_30pa["w_pd"] , label='Real part (30 Pa)')
# Normalized angular frequency
plt.xlabel('Wave number $k\lambda_D$')
plt.ylabel('Normalized angular frequency $\omega / \omega_{pd}$')
plt.title('Comparison of Theoretical and Experimental Dispersion Relations')
plt.legend()
plt.grid(True)
plt.show()

# If needed, you can also plot the imaginary part (damping term)
plt.figure(figsize=(8, 5))
plt.plot(k_values* params_20pa["debye_D"], omega_i_20pa/params_20pa["w_pd"], label='Imaginary part (20 Pa)', linestyle='--')
plt.plot(k_values* params_25pa["debye_D"], omega_i_25pa/params_25pa["w_pd"], label='Imaginary part (25 Pa)', linestyle='--')
plt.plot(k_values* params_30pa["debye_D"], omega_i_30pa/params_30pa["w_pd"], label='Imaginary part (30 Pa)', linestyle='--')
plt.xlabel('Wave number $k\lambda_D$')
plt.ylabel('Imaginary part of angular frequency $\omega_i$')
plt.title('Damping of Dust Acoustic Waves')
plt.legend()
plt.grid(True)
plt.show()
