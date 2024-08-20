#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:05:39 2024

@author: Luki
"""
import matplotlib.pyplot as plt
import numpy as np
import json

from scipy.integrate import quad as integrate
from scipy.optimize import fsolve
from scipy.constants import k, m_e, e, epsilon_0
from scipy.special import erf

import optuna
import optuna.visualization as vis
from functools import partial

data_v = json.load(open('Final-Results/finaldatastack.json'))

###############
#%% Constants #
###############

eV_K = 11606

sigma_neon = 10**(-18) #m^2

u = 1.660539066 *10**(-27)
m_neon = 20.1797 * u #*u = kg

def T_e_interpolation(x, I):
    C = [7.13, 7.06, 6.98, 5.5]
    D = [1.23, 0.75, 0.77, 1.59]
    y_data = np.add(C,np.divide(D,I))
    x_data = [20, 40, 60, 100]
    x_fit = np.linspace(15,30,100)
    #
    coef = np.polyfit(x_data,y_data,3)
    poly1d_fn = np.poly1d(coef)             # poly1d_fn is now a function which takes in x and returns an estimate for y
    #
    #fig, ax = plt.subplots(dpi=600)
    #fig.set_size_inches(4, 3)
    #ax.plot(x_fit, poly1d_fn(x_fit), linestyle='solid', color='#00cc00', linewidth=.7)
    #
    return poly1d_fn(x)

def n_e_interpolation(x, I):
    A = [1.92, 2.75, 3.15, 4.01]
    B = [-0.38, -0.42, -0.34, 0.047]
    y_data = np.add(np.multiply(A,I),np.multiply(B,I**2))
    x_data = [20, 40, 60, 100]
    x_fit = np.linspace(15,30,100)
    #
    coef = np.polyfit(x_data,y_data,1)
    poly1d_fn = np.poly1d(coef)             # poly1d_fn is now a function which takes in x and returns an estimate for y
    #
    #fig, ax = plt.subplots(dpi=600)
    #fig.set_size_inches(4, 3)
    #ax.plot(x_fit, poly1d_fn(x_fit), linestyle='solid', color='#00cc00', linewidth=.7)
    #
    return poly1d_fn(x)

def e_field(x,I):
    F = [1.97, 2.11, 2.07, 1.94]
    G = [0.14, 0.072, 0.098, 0.12]
    y_data = np.add(F,np.divide(G,I**2))
    x_data = [20, 40, 60, 100]
    x_fit = np.linspace(15,30,100)
    #
    coef = np.polyfit(x_data,y_data,1)
    poly1d_fn = np.poly1d(coef)             # poly1d_fn is now a function which takes in x and returns an estimate for y
    #
    #fig, ax = plt.subplots(dpi=600)
    #fig.set_size_inches(4, 3)
    #ax.plot(x_fit, poly1d_fn(x_fit), linestyle='solid', color='#00cc00', linewidth=.7)
    #
    return poly1d_fn(x)

#########################
#%% Variable Parameters #
#########################

I = .5 #mA

'''    Pressure    '''
p = np.array([15, 20, 25, 30, 40]) #pa

trial_nr = 0
trial = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

'''    Charge Potential    '''
#z = [.35, .34, .32, .32, .3] #F2
#z = [0.451, 0.32, 0.29, 0.29, 0.3] #F1
z = [.45, .35, .28, .24, .3] #Theory
z_inkl_error = [[0.545, 0.04],[0.397, 0.012],[0.377, 0.020],[0.378, 0.020],[0.345, 0.035]]

'''    Duty-Cycle    '''
dc_value = 1 

'''    Neutral damping Epstein coefficient 1. <= x <= 1.442    '''
epstein = [1.4, 1.4, 1.4, 1.4, 1.4]

'''    Particle Charge Depletion    '''
#charge_depletion = [0, 0, 0, 0, 1] #F1 & F2
charge_depletion = [0, 0, 0, 0, 0] #Theory

'''    Particle Size    '''
a = (1.3/2) *10**(-6) #micrometer particle radius

'''    Dust number density    '''
#n_d = [1.0, 1.0, 1.0, 1.0, 1.0] #F2
#n_d = [1.05, 1.3, 1.8, 2., 2.] #F1
n_d = [1.28, 2., 2., 2., 2.] #Theory
n_d = np.multiply(n_d,10**11) #in m^-3

##########################
# Calculate Parameters # Neon-Gas
##########################

'''    Electric Field    '''
ef_reduction = 1
E_0_calc = [e_field(15,I), e_field(20,I), e_field(25,I), e_field(30,I), e_field(40,I)]
E_0 = np.multiply(E_0_calc, -dc_value*100*ef_reduction)#-250*dc_value #210 V/m
E_0_vcm = np.divide(E_0, 100)

'''    Particle Temperatures     '''
T_e_reduction = 1
T_e = [T_e_interpolation(15,I), T_e_interpolation(20,I), T_e_interpolation(25,I), T_e_interpolation(30,I), T_e_interpolation(40,I)]
T_e = np.multiply(T_e, T_e_reduction)
T_e = T_e 
T_n = 0.025#eV

'''    Ion Mean free Path (Mittlere freie Weglänge, m^-4)    '''
E_0_multiplyer = np.array([0,0,0,0,0]) #F1
#E_0_multiplyer = np.array([1,1,0,0,0]) #F2
l_i = np.divide(T_n * k, p*sigma_neon)
T_i_tilde = np.multiply(2/9 * abs(np.multiply(E_0, E_0_multiplyer)) * e/k, l_i)
T_i = (T_i_tilde + 0.03)
T_iroom = [0.03, 0.03, 0.03, 0.03] #eV 

'''    Electron number density    '''
n_e0 = [n_e_interpolation(15,I), n_e_interpolation(20,I), n_e_interpolation(25,I), n_e_interpolation(30,I), n_e_interpolation(40,I)]
n_e0 = np.multiply(n_e0,10**14) #in m^-3

'''    Neutral Number Density    '''
n_0 = p/(k*T_n*11606)*10**(-6) #cm^-3
n_0_m = p/(k*T_n*11606) #m^-3
T_d = T_n#eV

'''    Dust mass and Neutral mass    '''
V = 4/3 * np.pi * a**3
roh = 1574 #kg/m^3
m_d = roh * V

#############
# Equations #
#############

'''    Neutral thermal temperature    '''
v_tn = np.sqrt(8 * k * T_n * 11606 / (np.pi * m_neon))

'''    Ion thermal temperature    '''
v_ti = np.sqrt(8 * k * T_i * 11606 / (np.pi * m_neon))

'''    Particle charge    '''
Z_d = 4 * np.pi * epsilon_0 * k * T_e * 11606 * a * z / (e**2)

n_i0 = np.add(n_e0, np.multiply(Z_d, n_d)) #m^-3 = n_e0 + Z_d*n_d ;melzer2019

'''    Debye electrons, ions and dust    '''
debye_De = np.sqrt(np.divide(epsilon_0 * k * T_e * 11606 , n_e0 * e**2))
debye_Di = np.sqrt(np.divide(epsilon_0 * k * T_i * 11606 , n_i0 * e**2))
debye_D  = np.divide(np.multiply(debye_De,debye_Di),np.sqrt(debye_De**2 + debye_Di**2))

'''    Scattering parameter (ion coupling param) https://doi.org/10.1063/1.1947027    '''
beta = np.divide(Z_d, debye_Di) * (e**2 / (4 * np.pi * epsilon_0 * m_neon * v_ti**2))

'''    Havnes Parameter    '''
P = np.multiply(np.multiply(695*(1.3/2),T_e),np.divide(n_d,n_i0))
P2 = np.multiply(Z_d/z,np.divide(n_d,n_i0))

'''    Charge depletion adjustment    '''
# New z from Havnes Parameter Physics Reports 421 (2005) 1 – 103#
def oml_func_p0(x):
    return np.sqrt(m_e/m_neon)*(1+x*tau[i]) - np.sqrt(tau[i]) * np.exp(-x)
def oml_func(x):
    return np.sqrt(m_e/m_neon)*(1+x*tau[i])*(1+P[i]) - np.sqrt(tau[i]) * np.exp(-x)
#
tau = np.divide(T_e,T_i)
n_de = np.divide(n_d,n_e0)
#
z_depl = []
if 1 in charge_depletion:   
    for i in range(len(charge_depletion)):
        if charge_depletion[i] == 1:
            root_p0 = fsolve(oml_func_p0, 0.4)
            root = fsolve(oml_func, 0.4)
            z_depl = np.append(z_depl,(((100 / root_p0) *root)/100) *z[i])
            Z_d[i] = ((4 * np.pi * epsilon_0 * k * T_e[i] * 11606 * a * z_depl[i]) / (e**2))
        else:
            z_depl = np.append(z_depl, z[i])
            Z_d[i] = ((4 * np.pi * epsilon_0 * k * T_e[i] * 11606 * a * z[i]) / (e**2))
    n_i0 = np.add(n_e0, np.multiply(Z_d, n_d)) #m^-3
    debye_De = np.sqrt(np.divide(epsilon_0 * k * T_e * 11606 , n_e0 * e**2))
    debye_Di = np.sqrt(np.divide(epsilon_0 * k * T_i * 11606 , n_i0 * e**2))
    debye_D  = np.divide(np.multiply(debye_De,debye_Di),np.sqrt(debye_De**2 + debye_Di**2))

'''    Modified Frost Formular    '''
A = 0.0321
B = 0.012
C = 1.181
EN = (-E_0_vcm/n_0)*(10**17) #10^17 from Vcm^2 to Td
M = A * np.abs((1 + np.abs((B * EN)**C))**(-1/(2*C))) * EN
v_ti2 = np.sqrt(k * T_i * 11606 / m_neon)
u_i = M*v_ti2 * dc_value

'''    Force Equations    '''
#
roh_0 = np.divide(Z_d , T_i*11606) * e**2 / (4 * np.pi * epsilon_0 * k)
#
beta_T = np.divide(Z_d * e**2, T_i*11606 * debye_Di * (4 * np.pi * epsilon_0 * k))
#
def function(x):
    return 2* np.exp(-x) * np.log((2*debye_Di[0]*x+roh_0[0])/(2*a*x+roh_0[0]))
def function1(x):
    return 2* np.exp(-x) * np.log((2*debye_Di[1]*x+roh_0[1])/(2*a*x+roh_0[1]))
def function2(x):
    return 2* np.exp(-x) * np.log((2*debye_Di[2]*x+roh_0[2])/(2*a*x+roh_0[2]))
def function3(x):
    return 2* np.exp(-x) * np.log((2*debye_Di[3]*x+roh_0[3])/(2*a*x+roh_0[3]))
def function4(x):
    return 2* np.exp(-x) * np.log((2*debye_Di[4]*x+roh_0[4])/(2*a*x+roh_0[4]))
integration_temp = integrate(function, 0, np.inf)[0]
integration_temp1 = integrate(function1, 0, np.inf)[0]
integration_temp2 = integrate(function2, 0, np.inf)[0]
integration_temp3 = integrate(function3, 0, np.inf)[0]
integration_temp4 = integrate(function4, 0, np.inf)[0]
#
integrated_f = np.array([integration_temp, integration_temp1, integration_temp2, integration_temp3, integration_temp4])
#
def function_bc(x):
    return 2* np.exp(-x) * np.log(1 + 2 * x / beta_T[0])
def function_bc1(x):
    return 2* np.exp(-x) * np.log(1 + 2 * x / beta_T[1])
def function_bc2(x):
    return 2* np.exp(-x) * np.log(1 + 2 * x / beta_T[2])
def function_bc3(x):
    return 2* np.exp(-x) * np.log(1 + 2 * x / beta_T[3])
def function_bc4(x):
    return 2* np.exp(-x) * np.log(1 + 2 * x / beta_T[4])
integration_bc = integrate(function_bc, 0, np.inf)[0]
integration_bc1 = integrate(function_bc1, 0, np.inf)[0]
integration_bc2 = integrate(function_bc2, 0, np.inf)[0]
integration_bc3 = integrate(function_bc3, 0, np.inf)[0]
integration_bc4 = integrate(function_bc4, 0, np.inf)[0]
#
integrated_bc = np.array([integration_bc, integration_bc1, integration_bc2, integration_bc3, integration_bc4])
#
zetta = M * np.sqrt(T_e / 2*T_i)
tau = T_e/T_i
r_c = a*np.array(z)*tau
ln_gamma = np.log(integrated_f)
#
aplha = 1 + 2*zetta**2 + (1 - 1 / (2*zetta**2)) * (1 + 2*(z*tau)) + 2*(np.array(z)**2*tau**2)*ln_gamma/zetta**2
beta = 1 + 2*(z*tau) + 2*zetta**2 - 4*np.array(z)**2*tau**2*ln_gamma
#
mu_id = (np.sqrt(np.pi) * a**2 * n_i0 * v_ti**2 / zetta) * (np.sqrt(np.pi/2) * erf(zetta) * (aplha) + (1 / (np.sqrt(2)*zetta)) * (beta) * np.exp(-zetta**2))

'''    Electric force    '''
F_e = Z_d * e * E_0

'''    Ion drag force Khrapak et. al. DOI: 10.1103/PhysRevE.66.046414 '''
F_i = np.multiply(n_i0,((8*np.sqrt(2*np.pi))/3) * m_neon * (v_ti2) * (u_i) * (a**2 + a*roh_0/2 +(roh_0**2) * integrated_f/4))
F_i2 = np.multiply(n_i0,((4*np.sqrt(2*np.pi))/3) * debye_Di**2 * m_neon * v_ti2 * beta_T**2 * u_i * integrated_bc)
F_i3 = mu_id * m_neon * u_i
'''    Particle velocity without ion drag force    '''
factor = np.multiply(epstein,(4/3)*np.pi*a**2*m_neon*v_tn*(p/(T_n*11606*k)))

'''    Particle velocity with ion drag force Khrapak et.al    '''
v_d = (F_e+F_i)/factor
v_d2 = (F_e+F_i2)/factor
v_d3 = abs((F_e+F_i3)/factor)

v_trial = []
if trial_nr < len(z):
    for i_trial in trial:
        F_e_trial = Z_d[trial_nr] * e * E_0[trial_nr] * i_trial
        EN = (-E_0_vcm*i_trial/n_0)*(10**17) #10^17 from Vcm^2 to Td
        M = A * np.abs((1 + np.abs((B * EN)**C))**(-1/(2*C))) * EN
        v_ti2 = np.sqrt(k * T_i * 11606 / m_neon)
        u_i = M*v_ti2 * dc_value
        F_i_trial = np.multiply(n_i0[trial_nr],((8*np.sqrt(2*np.pi))/3) * m_neon * (v_ti[trial_nr]) * (u_i[trial_nr]) * (a**2 + a*roh_0[trial_nr]/2 +(roh_0[trial_nr]**2) * integrated_f[trial_nr]/4))
        v_trial = np.append(v_trial, abs((F_e_trial+F_i_trial)/factor[trial_nr]))

'''    Plasma-Dust interaction frequency    '''
w_pd = np.sqrt(np.divide((np.multiply(Z_d**2,n_d*e**2)),(m_d*epsilon_0)))

'''    Damping rate beta    '''
beta_damp = np.multiply(p,epstein)*(8/(np.pi*a*roh*v_tn))    # with v_{th,n} = 320 m/s

'''    DAW - phase velocity    '''
v_0 = np.sqrt(2*(e*(Z_d))**2/(m_d*a))
aplha =(np.multiply(k*11606,T_i)/m_d)
epsilon = np.divide((n_d),(n_i0))
C_daw = np.sqrt(aplha * epsilon * Z_d**2) * 10**(3) #mm/s
C_daw_2 = w_pd * debye_Di * 10**(3) #mm/s

'''    Plasma-Ion interaction frequency    '''
w_pi = np.sqrt(4*np.pi*e**2*np.divide(n_i0,m_neon))

'''    Dust-Neutral collision frequency    '''
nu_dn = 8*(np.sqrt(2*np.pi)/3) * a**2 * v_tn * (m_neon/m_d) * np.divide(p,k*(T_n*11606))
'''    Ion-Neutral collision frequency    '''
nu_in = v_ti * sigma_neon * np.divide(p,k*(T_n))

'''    F_i / F_e ratio, Khrapak DOI: 10.1103/PhysRevE.66.046414     '''
beta_T = roh_0/debye_D
delta = (1/(3*np.sqrt(2*np.pi)))*(beta_T)*integrated_f
F_ie_ratio = delta * l_i / debye_D
#
#################################################
# PLOT
#################################################
#
v_group_100 = [data_v['15pa']['v_group_100'], data_v['20pa']['v_group_100'], data_v['25pa']['v_group_100'], data_v['30pa']['v_group_100'], data_v['40pa']['v_group_100']]
v_group_100_error = [data_v['15pa']['v_group_100_error'], data_v['20pa']['v_group_100_error'], data_v['25pa']['v_group_100_error'], data_v['30pa']['v_group_100_error'], data_v['40pa']['v_group_100_error']]
#
c_100 = [data_v['15pa']['c_daw_100'], data_v['20pa']['c_daw_100'], data_v['25pa']['c_daw_100'], data_v['30pa']['c_daw_100'], data_v['40pa']['c_daw_100']]
c_100_error = [data_v['15pa']['c_daw_100_error'], data_v['20pa']['c_daw_100_error'], data_v['25pa']['c_daw_100_error'], data_v['30pa']['c_daw_100_error'], data_v['40pa']['c_daw_100_error']]
#
fig, ax = plt.subplots(dpi=600)
fig.set_size_inches(4, 3)
ax.plot(p, abs(F_e)*10**(14), linestyle='solid', marker='^', color='#00429d', linewidth=.75)
ax.plot(p, F_i*10**(14), linestyle='solid', marker='x', color='#00cc00', linewidth=.75)
ax.legend(['$F_e x 10^{-14}$', '$F_i x 10^{-14}$'])
#adds major gridlines
ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
plt.show()
#
fig, ax = plt.subplots(dpi=600)
fig.set_size_inches(4, 3)
ax.errorbar(p, np.array(v_group_100)*(-1), yerr=v_group_100_error, fmt='^', color='#00429d', markersize=1, linewidth=.75, capsize=1)
ax.scatter(p[0:], v_d[0:]*1000, marker='x', linestyle='solid', color='#00cc00', linewidth=.7)
ax.legend(['Theory $v_{group}$', 'E100'])
#adds major gridlines
ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
plt.show()
#
fig, ax = plt.subplots(dpi=600)
fig.set_size_inches(4, 3)
ax.errorbar([15, 20, 25, 30, 40], c_100, yerr=c_100_error, fmt='^', color='#00429d', markersize=1, linewidth=.75, capsize=1)
ax.scatter(p, C_daw, linestyle='solid', marker='x', color='#00cc00', linewidth=.5)
ax.legend(['Theory $C_{DAW}$', 'E100'])
#adds major gridlines
ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
plt.show()
#
#%%
###############
#function to minimize
###############
#
#Global
global_vspeed = []
global_cspeed = []
global_error = []

def theory_v_c(z, n_d, pressure):
    '''Calculates particle velocity and speed of sound (C_daw) based on given parameters.'''
    Z_d = 4 * np.pi * epsilon_0 * k * T_e * 11606 * a * z / (e**2)
    n_i0 = np.add(n_e0, np.multiply(Z_d, n_d * 10**11))
    
    F_e = Z_d * e * E_0
    EN = (-E_0_vcm / n_0) * (10**17)  # Convert V/cm^2 to Td (Townsend)
    M = A * np.abs((1 + np.abs((B * EN)**C))**(-1 / (2 * C))) * EN
    v_ti2 = np.sqrt(k * T_i * 11606 / m_neon)
    u_i = M * v_ti2 * dc_value
    
    roh_0 = np.divide(Z_d, T_i * 11606) * e**2 / (4 * np.pi * epsilon_0 * k)
    
    integration_temp = integrate(function, 0, np.inf)[0]
    integration_temp1 = integrate(function1, 0, np.inf)[0]
    integration_temp2 = integrate(function2, 0, np.inf)[0]
    integration_temp3 = integrate(function3, 0, np.inf)[0]
    integration_temp4 = integrate(function4, 0, np.inf)[0]
    
    integrated_f = np.array([integration_temp, integration_temp1, integration_temp2, integration_temp3, integration_temp4])
    
    F_i = np.multiply(n_i0, ((8 * np.sqrt(2 * np.pi)) / 3) * m_neon * v_ti2 * u_i * (a**2 + a * roh_0 / 2 + (roh_0**2) * integrated_f / 4))
    
    v_dust = np.abs((F_e + F_i) / factor) * 1000
    alpha = (np.multiply(k * 11606, T_i) / m_d)
    epsilon = np.divide(n_d * 10**11, n_i0)
    C_daw = np.sqrt(alpha * epsilon * Z_d**2) * 10**3  # Convert to mm/s
    
    return v_dust.item(pressure), C_daw.item(pressure)

def objective(trial, pressure):
    try:
        # Parameter space
        z = trial.suggest_float('z', 0.2, 0.4)
        n_d = trial.suggest_float('n_d', 1., 2.)
        
        v_dust, c_daw = theory_v_c(z, n_d, pressure)

        expectation_v = abs(v_group_100[pressure])
        expectation_c = abs(c_100[pressure])
        verror = abs(expectation_v - v_dust)
        cerror = abs(expectation_c - c_daw)
        #print(verror + cerror)
        
        global_vspeed.append(v_dust)
        global_cspeed.append(c_daw)
        global_error.append(cerror + verror)

        return cerror + verror
    except Exception as e:
        print(f"Exception: {e}")
        return float('inf')

trials = 1000
pressure = 3  # Trial pressure

# Create a partial function to pass additional fixed parameters to the objective function
objective_partial = partial(objective, pressure=pressure)

# Bayesian optimization with optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective_partial, n_trials=trials)  # n_trials: number of iterations

# Extract best parameters and corresponding error from the created study
best_params = study.best_params
best_error = study.best_value

# Print the best parameters and error
print('----------------------------------------------------------')
print("Best Parameters:", best_params)
print("Best Error:", best_error)
print('----------------------------------------------------------')

# Visualization
optimization_history_plot = vis.plot_optimization_history(study)
optimization_history_plot.show()

param_importance_plot = vis.plot_param_importances(study)
param_importance_plot.show()

contour_plot = vis.plot_contour(study, params=["z", "n_d"])
contour_plot.show()
#
#END
#
#
#
'''    Critical Electric Field for DAWs    '''
constant_rsy = np.multiply(debye_Di,w_pd)
constant_rsy2 = np.sqrt(Z_d*((k*T_i*11606)/m_d))
#
E_crit = np.divide((k/e) * (T_i*11606/debye_Di),np.sqrt(np.add((w_pd/nu_dn)**2,(w_pi/nu_in)**2)))
#
E_crit2 = np.multiply((k*(T_i*11606)/e) , np.divide(nu_dn, C_daw*10**(-3))) #V/cm
#
epst_temp = np.divide((Z_d*e*E_0),np.multiply(m_d,C_daw*10**(-3)))
E_crit3 = np.divide(epst_temp,C_daw*10**(-3)) * (k*T_i*11606)/e