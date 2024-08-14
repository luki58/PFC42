# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:08:28 2024

@author: Lukas
"""

from __future__ import division, unicode_literals, print_function # Für die Kompatibilität mit Python 2 und 3.
import time
from tqdm import tqdm
startzeit = time.time()
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import SubplotDivider, Size, make_axes_locatable
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from tsmoothie.smoother import *
import pims
import scipy.ndimage as nd
import scipy.stats
from scipy.stats import t
from scipy.signal import find_peaks, hilbert, chirp, argrelmax
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, fsolve
import scipy.special as scsp
from scipy.integrate import quad as integrate, solve_ivp
from scipy import interpolate
from sklearn import linear_model
import codecs, json
from scipy.optimize import newton

data_v = json.load(open('Final-Results/finaldatastack.json'))

#############
# Constants #
#############

def k_b():
    return 1.38065 * 10**(-23)   #m^2kg/s^2K

def eps_0():
    return 8.854 * 10**(-12)   #As/Vm
    
def e():    
    return 1.6022 * 10**(-19)      #C

def sigma_neon():
    return 10**(-18) #m^2

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

#######################
# Variable Parameters #
#######################

I = .5 #mA

'''    Pressure    '''
p = np.array([15, 20, 25, 30, 40]) #pa

'''    Charge Potential    '''
z = [0.545, 0.395, 0.385, 0.385, 0.35]#0.35 #=0.3 +-0.1 for neon
z_inkl_error = [[0.545, 0.04],[0.397, 0.012],[0.377, 0.020],[0.378, 0.020],[0.345, 0.035]]

'''    Duty-Cycle    '''
dc_value = 1 

'''    Neutral damping Epstein coefficient 1. <= x <= 1.442    '''
epstein = [1.45, 1.45, 1.45, 1.45, 1.45]

'''    Particle Charge Depletion    '''
charge_depletion = True

'''    Particle Size    '''
a = (1.3/2) *10**(-6) #micrometer particle radius

'''    Dust number density    '''
n_d = [2., 2., 2., 2., 2.]
n_d = np.multiply(n_d,10**11) #in m^-3

########################
# Calculate Parameters # Neon-Gas
########################

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
T_i_reduction = 1
E_0_multiplyer = np.array([1,.4,.22,.22,0])
l_i = np.divide(T_n*k_b(),np.multiply(p,sigma_neon()))
T_i_tilde = np.multiply(2/9 * abs(np.multiply(E_0, E_0_multiplyer)) * e() / k_b(), l_i)
T_i = (T_i_tilde + 0.03)
#(np.abs(T_i_tilde) + 0.03*dc_value)*T_i_reduction
T_iroom = [0.03, 0.03, 0.03, 0.03] #eV 

'''    Electron number density    '''
n_e0 = [n_e_interpolation(15,I), n_e_interpolation(20,I), n_e_interpolation(25,I), n_e_interpolation(30,I), n_e_interpolation(40,I)]
n_e0 = np.multiply(n_e0,10**14) #in m^-3

'''    Neutral Number Density    '''
n_0 = p/(k_b()*T_n*11600)*10**(-6) #cm^-3
n_0_m = (p)/(k_b()*T_n*11600) #m^-3
T_d = T_n#eV

'''    Dust mass and Neutral mass    '''
V = 4/3 * np.pi * a**3
roh = 1574 #kg/m^3
m_d = roh * V
u = 1.660539066 *10**(-27)
m_neon = 20.1797 * u #*u = kg
m_e = 0.000548579909 * u #*u = kg

#############
# Equations #
#############

'''    Neutral thermal temperature    '''
v_tn = np.sqrt(8*k_b()*T_n*11600/(np.pi*m_neon))

'''    Dust thermal temperature    '''
v_td = v_tn

'''    Ion thermal temperature    '''
v_ti = np.sqrt(8*k_b()*T_i*11600/(np.pi*m_neon)) #(k_b*T_i/m_d)**(1/2) #particle thermal temperature

u_i = np.multiply(np.sqrt(k_b()/m_neon),T_e) #boom velocity

'''    Particle charge    '''
Z_d = []
for i in range(len(T_e)):
    Z_d = np.append(Z_d,((4*np.pi*eps_0()*k_b()*T_e[i]*11600*a*z[i])/(e()**2)))
    
'''    Ion number density    '''   
n_i0 = np.add(n_e0, np.multiply(Z_d, n_d)) #m^-3 = n_e0 + Z_d*n_d ;melzer2019

'''    Debye electrons, ions and dust    '''
debye_De = np.sqrt(np.divide(np.multiply((eps_0()*k_b()),np.multiply(T_e,11600)),np.multiply(n_e0,e()**2)))
debye_Di = np.sqrt(np.divide(np.multiply(eps_0()*k_b()*11600,T_i),np.multiply(n_i0,e()**2)))
debye_Dd = np.sqrt(np.divide(np.multiply(eps_0()*k_b(),v_td),np.multiply(n_d,e()**2)))
debye_D  = np.divide(np.multiply(debye_De,debye_Di),np.sqrt(debye_De**2 + debye_Di**2))
'''    Ion-Debye radius    '''
r_di = np.sqrt((eps_0()*k_b()*T_i)/(n_i0*e()**2))

'''    Scattering parameter (ion coupling param) https://doi.org/10.1063/1.1947027    '''
beta = np.divide(Z_d, debye_D) * (e()**2/(4*np.pi*eps_0()*m_neon*v_ti**2))

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
if charge_depletion == True:   
    for i in range(len(T_e)):
        root_p0 = fsolve(oml_func_p0, 0.4)
        root = fsolve(oml_func, 0.4)
        z_depl = np.append(z_depl,(((100 / root_p0) *root)/100) *z[i])
        Z_d[i] = ((4*np.pi*eps_0()*k_b()*T_e[i]*11600*a*z_depl[i])/(e()**2))
    n_i0 = np.add(n_e0, np.multiply(Z_d, n_d)) #m^-3 = n_e0 + Z_d*n_d ;melzer2019
    debye_De = np.sqrt(np.divide(np.multiply((eps_0()*k_b()),np.multiply(T_e,11600)),np.multiply(n_e0,e()**2)))
    debye_Di = np.sqrt(np.divide(np.multiply(eps_0()*k_b()*11600,T_i),np.multiply(n_i0,e()**2)))
    debye_Dd = np.sqrt(np.divide(np.multiply(eps_0()*k_b(),v_td),np.multiply(n_d,e()**2)))
    debye_D  = np.divide(np.multiply(debye_De,debye_Di),np.sqrt(debye_De**2 + debye_Di**2))
else:
    z_depl = np.ones(len(T_e))*z
    Z_d_0 = Z_d

'''    Modified Frost Formular    '''
A = 0.0321
B = 0.012
C = 1.181
EN = (-E_0_vcm/n_0)*(10**17) #????????????????????????????????????????????????????
M = A * np.abs((1 + np.abs((B*EN)**C))**(-1/(2*C))) * EN
v_ti2 = np.sqrt(k_b()*T_i*11600/m_neon)
u_i2 = M*v_ti2*(-1) * dc_value

'''    Force Equations    '''
#
roh_0 = np.divide(Z_d , T_i) * (e()**2/(4*np.pi*eps_0()*k_b()))
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


integrated_f = np.array([integration_temp, integration_temp1, integration_temp2, integration_temp3, integration_temp4])

dc_value2 = [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.]
            #0     1   2    3   4    5   6    7   8   9   10   11  12   13  14   15  16   17  18 
i = 1   
dc_value = 1
reduction_F_e = 1
reduction_F_i = 1
correction_ui = 1
correction_cdaw = [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.]

'''    Electric force    '''
F_e = -(E_0*e()*(Z_d)) * reduction_F_e
F_e_20 = np.multiply(-(E_0[i]*e()*(Z_d[i])), dc_value2)
'''    Ion drag force Khrapak et. al. DOI: 10.1103/PhysRevE.66.046414 '''
F_i = reduction_F_i * (10**-4)*np.multiply(n_e0,((8*np.sqrt(2*np.pi))/3) * m_neon * (v_ti) * (correction_ui * u_i2) * (a**2 + a*roh_0/2 +(roh_0**2) * integrated_f/4))
F_i_20 = np.multiply(dc_value2, (10**-4)*np.multiply(n_e0[i],((8*np.sqrt(2*np.pi))/3) * m_neon * (v_ti[i]) * (correction_ui *u_i2[i]) * (a**2 + a*roh_0[i]/2 +(roh_0[i]**2) * integrated_f[i]/4)))
'''    Particle velocity without ion drag force    '''
factor = (np.multiply(epstein,(4/3)*np.pi*a**2*m_neon*v_tn*(p/(T_n*11600*k_b()))))
factor_20 = (np.multiply(epstein[i],(4/3)*np.pi*a**2*m_neon*v_tn*(p[i]/(T_n*11600*k_b()))))
v_dust = np.divide(F_e,factor)


'''    Particle velocity with ion drag force Khrapak et.al    '''
v_dust_ink_iondrag = (F_e+F_i)/factor
v_dust_ink_iondrag_20 = (F_e_20 + F_i_20) / factor_20

'''    Plasma-Dust interaction frequency    '''
w_pd = np.sqrt(np.divide((np.multiply(Z_d**2,n_d*e()**2)),(m_d*eps_0())))

'''    Damping rate beta    '''
beta_damp = np.multiply(p,epstein)*(8/(np.pi*a*roh*v_tn))    # with v_{th,n} = 320 m/s

'''    DAW - phase velocity    '''
v_0 = np.sqrt(2*(e()*(Z_d))**2/(m_d*a))
aplha =(np.multiply(k_b()*11600,T_i)/m_d)
epsilon = np.divide((n_d),(n_i0))
C_daw = np.sqrt(aplha * epsilon * Z_d**2) * 10**(3) #mm/s
C_daw_2 = w_pd * debye_Di * 10**(3) #mm/s
aplha =(np.multiply(k_b()*11600,T_i[i])/m_d)
epsilon = np.divide((n_d[i]),(n_i0[i]))
C_daw_red = np.multiply(correction_cdaw, np.sqrt(aplha * epsilon * Z_d[i]**2) * 10**(3)) #mm/s 

'''    Plasma-Ion interaction frequency    '''
w_pi = np.sqrt(4*np.pi*e()**2*np.divide(n_i0,m_neon))

'''    Dust-Neutral collision frequency    '''
nu_dn = 8*(np.sqrt(2*np.pi)/3) * a**2 * v_tn * (m_neon/m_d) * np.divide(p,k_b()*(T_n*11600))
'''    Ion-Neutral collision frequency    '''
nu_in = v_ti * sigma_neon() * np.divide(p,k_b()*(T_n))

'''    F_i / F_e ratio, Khrapak DOI: 10.1103/PhysRevE.66.046414     '''
beta_2 = roh_0/debye_D
delta = (1/(3*np.sqrt(2*np.pi)))*(beta_2)*integrated_f
F_ie_ratio = delta * l_i / debye_D

'''    Critical Electric Field for DAWs    '''
constant_rsy = np.multiply(debye_Di,w_pd)
constant_rsy2 = np.sqrt(Z_d*((k_b()*T_i*11600)/m_d))
#
E_crit = np.divide((k_b()/e()) * (T_i*11600/debye_Di),np.sqrt(np.add((w_pd/nu_dn)**2,(w_pi/nu_in)**2)))
#
E_crit2 = np.multiply((k_b()*(T_i*11600)/e()) , np.divide(nu_dn, C_daw*10**(-3))) #V/cm
#
epst_temp = np.divide((Z_d*e()*E_0),np.multiply(m_d,C_daw*10**(-3)))
E_crit3 = np.divide(epst_temp,C_daw*10**(-3)) * (k_b()*T_i*11600)/e()

'''    ID Lattice c_s calculations    '''
M_correction = 1
#
interparticle_distance = 1/(
   n_d)**(1/3)
kappa = interparticle_distance/debye_D
c_0 = np.sqrt((Z_d*e())**2/(4*np.pi*eps_0()*interparticle_distance*m_d))
c_y = np.sqrt(c_0**2 * (((kappa*np.exp(kappa) *(kappa - 2 + 2* np.exp(kappa)))/(np.exp(kappa) -1)**2)- 2*np.log(np.exp(kappa) -1)))
c_d = 12.38 * c_0**2 *((M * M_correction)/kappa)**2
#
c_s = np.sqrt(c_y**2 + c_d**2)
#
S = np.exp(-kappa) * (1 + kappa + (kappa**2/2)) - 258 * (M * M_correction)**2 / (50 * kappa**2)

# PLOT

v_group_100 = [data_v['15pa']['v_group_100'], data_v['20pa']['v_group_100'], data_v['25pa']['v_group_100'], data_v['30pa']['v_group_100'], data_v['40pa']['v_group_100']]
v_group_100_error = [data_v['15pa']['v_group_100_error'], data_v['20pa']['v_group_100_error'], data_v['25pa']['v_group_100_error'], data_v['30pa']['v_group_100_error'], data_v['40pa']['v_group_100_error']]
#
c_100 = [data_v['15pa']['c_daw_100'], data_v['20pa']['c_daw_100'], data_v['25pa']['c_daw_100'], data_v['30pa']['c_daw_100'], data_v['40pa']['c_daw_100']]
c_100_error = [data_v['15pa']['c_daw_100_error'], data_v['20pa']['c_daw_100_error'], data_v['25pa']['c_daw_100_error'], data_v['30pa']['c_daw_100_error'], data_v['40pa']['c_daw_100_error']]
#
fig, ax = plt.subplots(dpi=600)
fig.set_size_inches(4, 3)
ax.errorbar([15, 20, 25, 30, 40], v_group_100, yerr=v_group_100_error, fmt='^', color='#00429d', markersize=1, linewidth=.75, capsize=1)
ax.scatter(p[:], v_dust_ink_iondrag[:]*1000, marker='x', linestyle='solid', color='#00cc00', linewidth=.7)
ax.legend(['E100','Theory $v_{group}$'])
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
#############
# Save Data #
#############
#
# Group Velocity #
v_dust = np.column_stack((v_dust,z_depl))
path = 'theo_ dustspeed_neutraldrag_dc' + str(int(dc_value*100)) + '_z' + str(round(np.average(z_depl), 3)) + 'adj_damping_1p15'
if charge_depletion == True:
    path = path + '_depleted.txt'
else: path = path + '.txt'
#
with open(path, 'w') as filehandle:
    json.dump(v_dust.tolist(), filehandle)
# Group Velocity #
v_dust_ink_iondrag = np.column_stack((v_dust_ink_iondrag,z_depl))
path = 'theo_ dustspeed_neutralandiondrag_dc' + str(int(dc_value*100)) + '_z' + str(round(np.average(z_depl), 3)) + 'adj_damping_1p15'
if charge_depletion == True:
    path = path + '_depleted.txt'
else: path = path + '.txt'
#
with open(path, 'w') as filehandle:
    json.dump(v_dust_ink_iondrag.tolist(), filehandle) 
# Dust-accoustic wave velocity linear #
C_daw_2 = np.column_stack((C_daw_2,z_depl))
path = 'theo_cdaw_dc100_z' + str(round(np.average(z_depl), 3)) + 'adj_damping_1p15' + '.txt'
with open(path, 'w') as filehandle:
    json.dump(C_daw_2.tolist(), filehandle)
#%% 
#
# Electric-Field depletion #
path = 'theo_v_group_20pa_ef-reduce.txt'
with open(path, 'w') as filehandle:
    json.dump(v_dust_ink_iondrag_20.tolist(), filehandle)
#%%
path = 'theo_v_group_ef30_pa-reduce.txt'
with open(path, 'w') as filehandle:
    json.dump(v_dust_ink_iondrag.tolist(), filehandle)
#%%
path = 'theo_c_daw_40pa_ef-reduce_linear.txt'
with open(path, 'w') as filehandle:
    json.dump(C_daw_red.tolist(), filehandle)
#%%
#
# Write System Parameters Json #
data = {
        "neon" : { "cross-section" : sigma_neon()
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
            "u_i" : u_i2[0]
            
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
            "u_i" : u_i2[1]    
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
            "u_i" : u_i2[2]
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
            "u_i" : u_i2[3]
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
            "u_i" : u_i2[4]
    }
}
with open('resultsC17/parameters/system-parameter-C15-230125.json', 'w') as filehandle:
    json.dump(data, filehandle)

#%%
i = 3
# Define the function to calculate a and b
def calculate_a_b(q):
    gamma_d = 1
    a = gamma_d * k_b() * T_d / m_d
    b = (epsilon * Z_d[i]**2 * k_b() * T_i[i] / m_d) * 1 / (1 + tau[i] * (1 - epsilon * Z_d[i]) + q**2 * debye_Di[i]**2)
    return a, b

# Define the dispersion relation function to solve for w
def dispersion_relation_eq(w, q, a, b):
    return w**2 + 1j * beta[i] * w - q**2 * (a + b)

# Define a range of q values
q_values = np.linspace(0.1, 10, 400)

# Calculate the dispersion relation for each q value
w_solutions = np.zeros_like(q_values, dtype=complex)

for i, q in enumerate(q_values):
    a, b = calculate_a_b(q)
    initial_guess = 1.0 + 1j  # Initial guess for w
    sol = root(lambda w: dispersion_relation_eq(w[0] + 1j * w[1], q, a, b), [initial_guess.real, initial_guess.imag], tol=1e-9)
    w_solutions[i] = sol.x[0] + 1j * sol.x[1]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(q_values, w_solutions.real, label='Real part of w')
plt.plot(q_values, w_solutions.imag, label='Imaginary part of w')
plt.xlabel('q')
plt.ylabel('w')
plt.title('Dispersion Relation: w vs q')
plt.legend()
plt.grid(True)
plt.show()
#end
