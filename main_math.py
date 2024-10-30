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
# Constants #
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
# Variable Parameters #
#########################

I = .5 #mA

'''    Pressure    '''
p = np.array([15, 20, 25, 30, 40]) #pa

trial_nr = 4
trial = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

'''    Charge Potential    '''
z = [0.54, 0.43, 0.42, 0.41, 0.32] #Exp F1
#z = [.3, .3, .3, .3, .3] #Theory1
z_inkl_error = [[0.54, 0.05],[0.43, 0.03],[0.42, 0.02],[0.41, 0.02],[0.32, 0.03]]

'''    Duty-Cycle    '''
dc_value = 1 

'''    Neutral damping Epstein coefficient 1. <= x <= 1.442    '''
epstein = [1.4, 1.4, 1.4, 1.4, 1.4]

'''    Particle Charge Depletion    '''
#charge_depletion = [0, 0, 0, 0, 0] #Theory
charge_depletion = [1, 1, 1, 1, 1] #Exp F1

'''    Particle Size    '''
a = (1.3/2) *10**(-6) #micrometer particle radius

'''    Dust number density    '''
n_d = [.65, 1.5, 1.9, 2.3, 2.5] #Exp^F1
#n_d = [1., 1., 1., 1., 1.] #Theory1
#n_d = [2., 2., 2., 2., 2.] #Theory2
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
E_0_multiplyer = np.array([1,1,1,1,1]) #F1
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
Z_d_0 = 4 * np.pi * epsilon_0 * k * T_e * 11606 * a * z / (e**2)

n_i0 = np.add(n_e0, np.multiply(Z_d_0, n_d)) #m^-3 = n_e0 + Z_d*n_d ;melzer2019

'''    Debye electrons, ions and dust    '''
debye_De = np.sqrt(np.divide(epsilon_0 * k * T_e * 11606 , n_e0 * e**2))
debye_Di = np.sqrt(np.divide(epsilon_0 * k * T_i * 11606 , n_i0 * e**2))
debye_D  = np.divide(np.multiply(debye_De,debye_Di),np.sqrt(debye_De**2 + debye_Di**2))

'''    Scattering parameter (ion coupling param) https://doi.org/10.1063/1.1947027    '''
beta = np.divide(Z_d_0, debye_Di) * (e**2 / (4 * np.pi * epsilon_0 * m_neon * v_ti**2))

'''    Havnes Parameter    '''
P = np.multiply(np.multiply(695*(1.3/2),T_e),np.divide(n_d,n_i0))
P2 = np.multiply(Z_d_0/z,np.divide(n_d,n_i0))

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
Z_d = Z_d_0
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

cos_teta = 0
cos_2_teta = np.array([.5, .6, .7, .8, .9])

F_bgk = debye_Di**2 * (-np.sqrt(8/np.pi) * u/v_tn * cos_teta + (2-np.pi/2) * u**2/v_tn**2 * (1 - 3 * cos_2_teta))

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
F_i = np.multiply(n_i0,((8*np.sqrt(2*np.pi))/3) * m_neon * (v_ti) * (u_i) * (a**2 + a*roh_0/2 +(roh_0**2) * integrated_f/4))
F_i2 = np.multiply(n_i0,((4*np.sqrt(2*np.pi))/3) * debye_Di**2 * m_neon * v_ti2 * beta_T**2 * u_i * integrated_bc)
F_i3 = mu_id * m_neon * u_i
'''    Particle velocity without ion drag force    '''
factor = np.multiply(epstein,(4/3)*np.pi*a**2*m_neon*v_tn*(p/(T_n*11606*k)))

'''    Particle velocity with ion drag force Khrapak et.al    '''
v_d = (F_e+F_i)/factor
v_d2 = (F_e+F_i2)/factor
v_d3 = abs((F_e+F_i3)/factor)

''' Modeling Charge for Polarity switching trails '''
#%%
def poly_function(input_value, a1,b1,c1,d1):
    #Recale x from duty-cycle to eff eletric field 0 dc -> 1. efield; .5 dc -> 0 efield
    x = abs(.5 - input_value/2)
    #x = input_value
    # Calculate the value of the cubic polynomial
    #return (2-np.pi/2)*(1 - (a1 * x**3 + b1 * x**2 + c1 * x + d1))
    return abs(-(2-np.pi/2)*(3*(a1 * x**3 + b1 * x**2 + c1 * x + d1) - 1))
def poly_raw(input_value, a5,b5,c5,d5):
    x = abs(.5 - input_value/2)
    # Calculate the value of the cubic polynomial
    return abs(a1 * x**3 + b1 * x**2 + c1 * x + d1)

def objective(trial, trial_nr, data_trial, data_trial_error, data_trial_axis):
    try:
        # Parameter space
        da = 11
        a3 = trial.suggest_float('a', (-8.61111111)-da, (-8.61111111)+da)
        db = 9
        b2 = trial.suggest_float('b', (6.76190476)-db, (6.76190476)+db)
        dc = .3
        c1 = trial.suggest_float('c', (-0.32579365)-dc, (-0.32579365)+dc)
        dd = .4
        d0 = trial.suggest_float('d', (0.45119048)-dd, (0.45119048)+dd)

        v_trial = v_trial_function(trial_nr, a3, b2, c1, d0)
        verror = 0
        for i in range(len(data_trial_axis)):
            index = int(data_trial_axis[i] * 10)
            if index < len(v_trial):
                expected_value = data_trial[i] / 1000
                error_margin = data_trial_error[i] / 1000
                # Fit within the data range considering the error
                if v_trial[index] < expected_value - error_margin or v_trial[index] > expected_value + error_margin:
                    verror += abs(expected_value - v_trial[index]) - error_margin
                else:
                    verror += 0 #in the rror coridor
                
        return verror

    except Exception as e:
        print(f"Exception: {e}")
        return float('inf')

def v_trial_function(trial_nr, a1,b1,c1,d1):
    v_trial = []
    if trial_nr < len(z):
        for i_trial in trial:
            #
            Z_d_new = Z_d[trial_nr] * poly_function(i_trial/2, a1,b1,c1,d1)
            F_e_trial = Z_d_new * e * E_0[trial_nr] * i_trial
            EN = (-E_0_vcm*i_trial/n_0)*(10**17) #10^17 from Vcm^2 to Td
            M = A * np.abs((1 + np.abs((B * EN)**C))**(-1/(2*C))) * EN
            v_ti2 = np.sqrt(k * T_i * 11606 / m_neon)
            u_i = M*v_ti2 * dc_value
            #
            '''    Force Equations    '''
            #
            roh_0 = np.divide(Z_d_new , T_i*11606) * e**2 / (4 * np.pi * epsilon_0 * k)
            #
            beta_T = np.divide(Z_d_new * e**2, T_i*11606 * debye_Di * (4 * np.pi * epsilon_0 * k))
            #
            integration_temp = integrate(function, 0, np.inf)[0]
            integration_temp1 = integrate(function1, 0, np.inf)[0]
            integration_temp2 = integrate(function2, 0, np.inf)[0]
            integration_temp3 = integrate(function3, 0, np.inf)[0]
            integration_temp4 = integrate(function4, 0, np.inf)[0]
            #
            integrated_f = np.array([integration_temp, integration_temp1, integration_temp2, integration_temp3, integration_temp4])
            #
            F_i_trial = np.multiply(n_i0[trial_nr],((8*np.sqrt(2*np.pi))/3) * m_neon * (v_ti[trial_nr]) * (u_i[trial_nr]) * (a**2 + a*roh_0[trial_nr]/2 +(roh_0[trial_nr]**2) * integrated_f[trial_nr]/4))
            v_trial = np.append(v_trial, abs((F_e_trial+F_i_trial)/factor[trial_nr]))
            #
    return v_trial
#
trial_data_20 = np.array([data_v['20pa']['v_group_30'], data_v['20pa']['v_group_40'], data_v['20pa']['v_group_60'], data_v['20pa']['v_group_80'], data_v['20pa']['v_group_100']])
trial_data_20_error = np.array([data_v['20pa']['v_group_30_error'], data_v['20pa']['v_group_40_error'], data_v['20pa']['v_group_60_error'], data_v['20pa']['v_group_80_error'], data_v['20pa']['v_group_100_error']])
trial_data_25 = np.array([data_v['25pa']['v_group_30'], data_v['25pa']['v_group_40'], data_v['25pa']['v_group_60'], data_v['25pa']['v_group_80'], data_v['25pa']['v_group_100']])
trial_data_25_error = np.array([data_v['25pa']['v_group_30_error'], data_v['25pa']['v_group_40_error'], data_v['25pa']['v_group_60_error'], data_v['25pa']['v_group_80_error'], data_v['25pa']['v_group_100_error']])
trial_data_30 = np.array([data_v['30pa']['v_group_40'], data_v['30pa']['v_group_50'], data_v['30pa']['v_group_60'], data_v['30pa']['v_group_80'], data_v['30pa']['v_group_100']])
trial_data_30_error = np.array([data_v['30pa']['v_group_40_error'], data_v['30pa']['v_group_50_error'], data_v['30pa']['v_group_60_error'], data_v['30pa']['v_group_80_error'], data_v['30pa']['v_group_100_error']])
trial_data_40 = np.array([data_v['40pa']['v_group_30'], data_v['40pa']['v_group_50'], data_v['40pa']['v_group_70'], data_v['40pa']['v_group_90'], data_v['40pa']['v_group_100']])
trial_data_40_error = np.array([data_v['40pa']['v_group_30_error'], data_v['40pa']['v_group_50_error'], data_v['40pa']['v_group_70_error'], data_v['40pa']['v_group_90_error'], data_v['40pa']['v_group_100_error']])
#
data_trial = [np.array([0]), trial_data_20, trial_data_25, trial_data_30, trial_data_40]
trial_data_40_error[-2] = trial_data_40_error[-2] -2
data_trial_error = [np.array([0]), trial_data_20_error, trial_data_25_error, trial_data_30_error, trial_data_40_error]
data_trial_axis = [[0], np.array([30, 40, 60, 80, 100])/100, np.array([30, 40, 60, 80, 100])/100, np.array([40, 50, 60, 80, 100])/100, np.array([30, 50, 70, 90, 100])/100]
#
# Optimization and plotting for trial_nr from 1 to 4
# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=600)
poly_legend = []
#
for trial_nr in range(1, 5):
    objective_partial = partial(objective, trial_nr=trial_nr, data_trial=data_trial[trial_nr], data_trial_error=data_trial_error[trial_nr], data_trial_axis=data_trial_axis[trial_nr])


    # Bayesian optimization with optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_partial, n_trials=800)

    # Extract best parameters from the created study
    best_params = study.best_params
    a1, b1, c1, d1 = best_params['a'], best_params['b'], best_params['c'], best_params['d']

    # Generate input values for the plot
    input_values = np.linspace(0, 1, 100)
    polynomial_values = [poly_raw(val, a1, b1, c1, d1) for val in input_values]
    
    # Create a string for the polynomial function to display in the legend
    poly_legend.append(f"$f_{trial_nr}(x)$ = ${a1:.2f}x^3 + {b1:.2f}x^2 {c1:.2f}x + {d1:.2f}$")
    
    color_codes = ['#D81B1B', '#48A2F1', '#FFC107', '#004D40']
    fmt_codes = ['s', '^', 'o', 'd']
    linestyle_codes = ['dotted', 'dashed', 'dashdot', 'solid']

    # First subplot: Scatter plot of trial vs v_trial_function
    ax1.plot(np.array(trial), v_trial_function(trial_nr, a1, b1, c1, d1), linestyle=linestyle_codes[trial_nr-1], color=color_codes[trial_nr-1], linewidth=.8)
    #ax1.scatter(data_trial_axis[trial_nr], data_trial[trial_nr] / 1000, marker='o', linestyle='solid', color=color_codes[trial_nr-1], linewidth=.7)
    ax1.errorbar(data_trial_axis[trial_nr], data_trial[trial_nr] / 1000, yerr=data_trial_error[trial_nr] / 1000, fmt=fmt_codes[trial_nr-1], color=color_codes[trial_nr-1], markersize=3, linewidth=.8, capsize=1, mfc='w') 
    #ax1.set_title(f"v_trial_function vs trial (Trial #{trial_nr})")
    ax1.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)

    # Second subplot: Polynomial plot
    ax2.plot(np.array(input_values), polynomial_values, label="Polynomial", color=color_codes[trial_nr-1], linestyle=linestyle_codes[trial_nr-1], linewidth=.8)
    #ax2.set_title(f"Polynomial Function (Trial #{trial_nr})")
    ax2.grid(color='grey', linestyle=linestyle_codes[trial_nr-1], linewidth=0.4, alpha=0.5)
# 
# Configure ax1 top x-axis for duty-cycle
ax2_top = ax2.twiny()  # Create a twin x-axis sharing the y-axis with ax2
ax2_top.set_xlim(ax2.get_xlim())  # Sync limits with ax2
duty_cycle_values = [60, 50, 40, 30, 20, 10, 0]
ax2_top.set_xticklabels([f"{int(dc)}" for dc in duty_cycle_values])  # Labels as duty cycle
ax2_top.set_xlabel("Duty-Cycle [%]")
#
# Adjust E_eff labels on x-axis to show percentage format
ax1.set_xticklabels([f"{int(x*100)}" for x in ax1.get_xticks()])
ax2.set_xticklabels([f"{int(x*100)}" for x in ax2.get_xticks()])
#
# Adjust v_group labels on y-axis to show values *1000
ax1.set_yticklabels([f"{int(y*1000)}" for y in ax1.get_yticks()])
#
# Legend and other labels remain the same
ax1.legend(['$v^{20Pa} (f_{charge})$', '$v^{25Pa} (f_{charge})$', '$v^{30Pa} (f_{charge})$', '$v^{40Pa} (f_{charge})$'], loc='upper left')
ax1.set_xlabel("$E_{eff}$ [%]")
ax1.set_ylabel("$v_{group}$ [mm/s]")
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_ylabel("$f_{charge}$")
ax2.set_xlabel("$E_{eff}$ [%]")
#
plt.tight_layout()
plt.show()
#
print(poly_legend)
#
#%%
#trial_nr=4
#
a1= best_params['a']
b1= best_params['b']
c1= best_params['c']
d1= best_params['d']
#
# Generate input values for the plot
input_values = np.linspace(0, 1, 100)
polynomial_values = [poly_function(val, a1, b1, c1, d1) for val in input_values]
#
# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=600)
# First subplot: Scatter plot of trial vs v_trial_function
ax1.scatter(trial, v_trial_function(trial_nr, a1, b1, c1, d1), marker='x', linestyle='solid', color='#00cc00', linewidth=.7)
ax1.scatter(data_trial_axis[trial_nr], data_trial[trial_nr] / 1000, marker='o', linestyle='solid', color='#000000', linewidth=.7)
ax1.set_title("v_trial_function vs trial")
ax1.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)

# Second subplot: Polynomial plot
ax2.plot(input_values, polynomial_values, label="Polynomial", color='#00cc00')
ax2.set_title("Polynomial Function")
ax2.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)

# Show the plots side by side
plt.tight_layout()
plt.show()
#%%
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

# Dispersion relation function
def dispersion_relation(w, k):
    chi_e = 1 / (k * debye_De)**2
    chi_i = w_pi**2 / (k**2 * v_ti2**2 + k * u_i *(1j * nu_in - k * u_i))
    chi_d = -w_pd**2 / (w + k*np.array(v_group_100)) * (w + k*np.array(v_group_100) + 1j*nu_in)  # Assuming dust-neutral collision frequency is 0.1 * nu_in
    return 1 + chi_e + chi_i + chi_d

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
ax.errorbar(p, np.array(v_group_100), yerr=v_group_100_error, fmt='^', color='#00429d', markersize=1, linewidth=.75, capsize=1)
ax.scatter(p[0:], v_d[0:]*(-1000), marker='x', linestyle='solid', color='#00cc00', linewidth=.7)
ax.legend(['Theory1 $v_{group}$', 'E100'])
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
#%%
# Group Velocity #
v_d = np.column_stack((v_d,z_depl))
path = 'theo_dustspeed_neutralandiondrag_dc' + str(int(dc_value*100)) + '_z' + str(round(np.average(z_depl), 3))
if 1 in charge_depletion:
    path = path + '_depleted.txt'
else: path = path + '.txt'
with open(path, 'w') as filehandle:
    json.dump(v_d.tolist(), filehandle) 
# Dust-accoustic wave velocity linear #
C_daw_2 = np.column_stack((C_daw_2,z_depl))
path = 'theo_cdaw_dc100_z' + str(round(np.average(z_depl), 3)) + '.txt'
with open(path, 'w') as filehandle:
    json.dump(C_daw_2.tolist(), filehandle)
# Electric-Field depletion #
path = 'theo_v_group_40pa_ef-reduce.txt'
with open(path, 'w') as filehandle:
    json.dump(v_trial.tolist(), filehandle)
#%% Write System Parameters Json #
data = {
        "neon" : { "cross-section" : sigma_neon
    },
        "15pa" : {
            "w_pd" : w_pd[0],
            "w_pi" : w_pi[0],
            "debye_De" : debye_De[0],
            "debye_Di" : debye_Di[0],
            "debye_D" : debye_D[0],
            "beta_damp" : beta_damp[0],
            "havnes" : P[0],
            "e-field-vm" : E_0[0],
            "Z_d_0" : Z_d_0[0],
            "Z_d" : Z_d[0],
            "z" : z_inkl_error[0][0],
            "z_error" : z_inkl_error[0][1],
            "z_depl" : z_depl[0],
            "l_i" : l_i[0],
            "n_d" : n_d[0],
            "n_i" : n_i0[0],
            "n_e" : n_e0[0],
            "T_i" : T_i[0],
            "T_e" : T_e[0],
            "n_0" : n_0_m[0],
            "u_i" : u_i[0]
            
    },
        "20pa" : {
            "w_pd" : w_pd[1],
            "w_pi" : w_pi[1],
            "debye_De" : debye_De[1],
            "debye_Di" : debye_Di[1],
            "debye_D" : debye_D[1],
            "beta_damp" : beta_damp[1],
            "havnes" : P[1] ,
            "e-field-vm" : E_0[1],
            "Z_d_0" : Z_d_0[1],
            "Z_d" : Z_d[1],
            "z" : z_inkl_error[1][0],
            "z_error" : z_inkl_error[1][1],
            "z_depl" : z_depl[1],
            "l_i" : l_i[1],
            "n_d" : n_d[1],
            "n_i" : n_i0[1],
            "n_e" : n_e0[1],
            "T_i" : T_i[1],
            "T_e" : T_e[1],
            "n_0" : n_0_m[1],
            "u_i" : u_i[1]    
    },
        "25pa" : {
            "w_pd" : w_pd[2],
            "w_pi" : w_pi[2],
            "debye_De" : debye_De[2],
            "debye_Di" : debye_Di[2],
            "debye_D" : debye_D[2],
            "beta_damp" : beta_damp[2],
            "havnes" : P[2],
            "e-field-vm" : E_0[2],
            "Z_d_0" : Z_d_0[2],
            "Z_d" : Z_d[2],
            "z" : z_inkl_error[2][0],
            "z_error" : z_inkl_error[2][1],
            "z_depl" : z_depl[2],
            "l_i" : l_i[2],
            "n_d" : n_d[2],
            "n_i" : n_i0[2],
            "n_e" : n_e0[2],
            "T_i" : T_i[2],
            "T_e" : T_e[2],
            "n_0" : n_0_m[2],
            "u_i" : u_i[2]
    },
        "30pa" : {
            "w_pd" : w_pd[3],
            "w_pi" : w_pi[3],
            "debye_De" : debye_De[3],
            "debye_Di" : debye_Di[3],
            "debye_D" : debye_D[3],
            "beta_damp" : beta_damp[3],
            "havnes" : P[3],
            "e-field-vm" : E_0[3],
            "Z_d_0" : Z_d_0[3],
            "Z_d" : Z_d[3],
            "z" : z_inkl_error[3][0],
            "z_error" : z_inkl_error[3][1],
            "z_depl" : z_depl[3],
            "l_i" : l_i[3],
            "n_d" : n_d[3],
            "n_i" : n_i0[3],
            "n_e" : n_e0[3],
            "T_i" : T_i[3],
            "T_e" : T_e[3],
            "n_0" : n_0_m[3],
            "u_i" : u_i[3]
    },
        "40pa" : {
            "w_pd" : w_pd[4],
            "w_pi" : w_pi[4],
            "debye_De" : debye_De[4],
            "debye_Di" : debye_Di[4],
            "debye_D" : debye_D[4],
            "beta_damp" : beta_damp[4],
            "havnes" : P[4],
            "e-field-vm" : E_0[4],
            "Z_d_0" : Z_d_0[4],
            "Z_d" : Z_d[4],
            "z" : z_inkl_error[4][0],
            "z_error" : z_inkl_error[4][1],
            "z_depl" : z_depl[4],
            "l_i" : l_i[4],
            "n_d" : n_d[4],
            "n_i" : n_i0[4],
            "n_e" : n_e0[4],
            "T_i" : T_i[4],
            "T_e" : T_e[4],
            "n_0" : n_0_m[4],
            "u_i" : u_i[4]
    }
}
with open('resultsC17/parameters/system-parameter-C15-230125.json', 'w') as filehandle:
    json.dump(data, filehandle)
#%%
forces = {
    "15pa" : {
        "F_i" : F_i[0],
        "F_e" : F_e[0],
        "n_d" : n_d[0]
        
        },
    "20pa" : {
        "F_i" : F_i[1],
        "F_e" : F_e[1],
        "n_d" : n_d[1]
        },
    "25pa" : {
        "F_i" : F_i[2],
        "F_e" : F_e[2],
        "n_d" : n_d[2]
        },
    "30pa" : {
        "F_i" : F_i[3],
        "F_e" : F_e[3],
        "n_d" : n_d[3]
        },
    "40pa" : {
        "F_i" : F_i[4],
        "F_e" : F_e[4],
        "n_d" : n_d[4]
        }
}
with open('Final-Results/forces.json', 'w') as filehandle:
    json.dump(forces, filehandle)
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
charge_depletion = [1, 1, 1, 1, 1]
#
# Example functions that need proper definitions
def oml_func_p0(x, tauf):
    return np.sqrt(m_e / m_neon) * (1 + x * tauf) - np.sqrt(tauf) * np.exp(-x)

def oml_func(x, Pf, tauf):
    return np.sqrt(m_e / m_neon) * (1 + x * tauf) * (1 + Pf) - np.sqrt(tauf) * np.exp(-x)

def deplete_z(z1, Pf, tauf):
    z_deplet = np.ones(len(charge_depletion))  # Preallocate the array with the same shape as z
    for i in range(len(charge_depletion)):
        if charge_depletion[i] == 1:
            root_p0 = fsolve(oml_func_p0, 0.4, args=(tauf[i]))[0]
            root = fsolve(oml_func, 0.4, args=(Pf[i], tauf[i]))[0]
            value = ((((100 / root_p0) * root) / 100) * z1)
            z_deplet[i] = value  # Assign the float directly
        else:
            z_deplet[i] = z1
    return z_deplet
#
def theory_v_c(z1, n_d_f, pressure):
    '''Calculates particle velocity and speed of sound (C_daw) based on given parameters.'''
    Z_d = 4 * np.pi * epsilon_0 * k * T_e * 11606 * a * z1 / (e**2)
    n_d = n_d_f*10**11
    tauf = np.divide(T_e,T_i)
    n_i0 = np.add(n_e0, np.multiply(Z_d, n_d))
    Pf = np.multiply(Z_d/z1,np.divide(n_d,n_i0))
    z_deplf = deplete_z(z1,Pf,tauf)
    Z_d_new = 4 * np.pi * epsilon_0 * k * T_e * 11606 * a * z_deplf / (e**2)
    #
    n_i0 = np.add(n_e0, np.multiply(Z_d_new, n_d)) #m^-3
    #
    F_e = Z_d_new * e * E_0
    EN = (-E_0_vcm / n_0) * (10**17)  # Convert V/cm^2 to Td (Townsend)
    M = A * np.abs((1 + np.abs((B * EN)**C))**(-1 / (2 * C))) * EN
    v_ti2 = np.sqrt(k * T_i * 11606 / m_neon)
    u_i = M * v_ti2 * dc_value
    
    roh_0 = np.divide(Z_d_new, T_i * 11606) * e**2 / (4 * np.pi * epsilon_0 * k)
    
    integration_temp = integrate(function, 0, np.inf)[0]
    integration_temp1 = integrate(function1, 0, np.inf)[0]
    integration_temp2 = integrate(function2, 0, np.inf)[0]
    integration_temp3 = integrate(function3, 0, np.inf)[0]
    integration_temp4 = integrate(function4, 0, np.inf)[0]
    
    integrated_f = np.array([integration_temp, integration_temp1, integration_temp2, integration_temp3, integration_temp4])
    
    F_i = np.multiply(n_i0, ((8 * np.sqrt(2 * np.pi)) / 3) * m_neon * v_ti * u_i * (a**2 + a * roh_0 / 2 + (roh_0**2) * integrated_f / 4))
    
    v_dust = ((F_e + F_i) / factor) * (-1000)
    alpha = (np.multiply(k * 11606, T_i) / m_d)
    epsilon = np.divide(n_d, n_i0)
    C_daw = np.sqrt(alpha * epsilon * np.array(Z_d)**2) * 10**3  # Convert to mm/s
    
    return v_dust.item(pressure), C_daw.item(pressure)
    

def objective(trial, pressure):
    try:
        # Parameter space
        z = trial.suggest_float('z', 0.4, 0.7)
        n_d = trial.suggest_float('n_d', 1., 2.3)
        
        v_dust, c_daw = theory_v_c(z, n_d, pressure)

        expectation_v = v_group_100[pressure]
        expectation_c = c_100[pressure]
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

trials = 100
pressure = 1  # Trial pressure

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

#%% Visualization
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