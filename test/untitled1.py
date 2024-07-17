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
u = 1.660539066 *10**(-27)
m_neon = 20.1797 * u #*u = kg
k_B = k  # Boltzmann constant in J/K


# Load system parameters
with open('C://Users/Lukas/Documents/GitHub/PFC42/resultsC17/parameters/system-parameter-C15-230125.json', 'r') as file:
    system_params = json.load(file)
# Load final data stack
with open('C://Users/Lukas/Documents/GitHub/PFC42/Final-Results/finaldatastack.json', 'r') as file:
    final_data = json.load(file)

# Given parameters for the plasma
params_20pa = system_params["20pa"]
params_25pa = system_params["25pa"]
params_30pa = system_params["30pa"]
#
fd_20pa = final_data["20pa"]
fd_25pa = final_data["25pa"]
fd_30pa = final_data["30pa"]

# Wave number range
k_values = np.linspace(0.01, 2500, 5000)  # in 1/m, adjusted for the typical range of kλ_D

def solve_dispersion_relation_simplified(params, fd):
    omega_r = []
    omega_i = []
    for k_val in k_values:
        # Solve the simplified dispersion relation for omega
        def dispersion_relation(omega):
            return omega**2 + 1j * params["beta_damp"] * omega - (params["w_pd"] * k_val**2 * params["debye_D"]) / (1 + k_val**2 * params["debye_D"]**2)
        
        def dispersion_relation_2(omega):
            vTi = np.sqrt(k_B*params["T_i"]*11600/(m_neon))
            nu_in = vTi / params["debye_Di"]  # Ion-neutral collision frequency
            mu_i = e / (m_neon * nu_in)  # Ion mobility
            ui = mu_i * abs(params["e-field-vm"])  # Electric field strength
            chi_e = 1 / (k * params['debye_De'])**2
            chi_i = params["w_pi"]**2 / (k_val**2 * vTi**2 + k_val * ui *(1j * nu_in - k_val * ui))
            chi_d = -params["w_pd"]**2 / (omega + k_val*fd["v_group_100"]) * (omega + k_val*fd["v_group_100"] + 1j*nu_in)  # Assuming dust-neutral collision frequency is 0.1 * nu_in
            return 1 + chi_e + chi_i + chi_d
        
        # Initial guess for omega
        omega_initial_guess = np.sqrt((params["w_pd"] * k_val**2 * params["debye_D"]) / (1 + k_val**2 * params["debye_D"]**2))
        
        # Solve for the real and imaginary parts of omega
        omega_sol = fsolve(lambda omega: np.real(dispersion_relation(omega)), omega_initial_guess)
        omega_imag_sol = np.imag(dispersion_relation(omega_sol[0]))
        
        # Combine the real and imaginary parts
        omega_r.append(omega_sol[0])
        omega_i.append(omega_imag_sol)
        
    return np.array(omega_r), np.array(omega_i)

# Solving the dispersion relation for each pressure
omega_r_20pa, omega_i_20pa = solve_dispersion_relation_simplified(params_20pa, fd_20pa)
omega_r_25pa, omega_i_25pa = solve_dispersion_relation_simplified(params_25pa, fd_25pa)
omega_r_30pa, omega_i_30pa = solve_dispersion_relation_simplified(params_30pa, fd_30pa)

# Plotting the theoretical dispersion relation
plt.figure(figsize=(8, 5), dpi=300)
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
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(k_values* params_20pa["debye_D"], omega_i_20pa/params_20pa["w_pd"], label='Imaginary part (20 Pa)', linestyle='--')
plt.plot(k_values* params_25pa["debye_D"], omega_i_25pa/params_25pa["w_pd"], label='Imaginary part (25 Pa)', linestyle='--')
plt.plot(k_values* params_30pa["debye_D"], omega_i_30pa/params_30pa["w_pd"], label='Imaginary part (30 Pa)', linestyle='--')
plt.xlabel('Wave number $k\lambda_D$')
plt.ylabel('Imaginary part of angular frequency $\omega_i / \omega_{pd}$')
plt.title('Damping of Dust Acoustic Waves')
plt.legend()
plt.grid(True)
plt.show()
