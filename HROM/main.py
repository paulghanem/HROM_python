# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:27:47 2023

@author: siliconsynapse
"""

from functions_HROM import *
from HROM_compliant_model import *
from HROM_simulation import *
import autograd.numpy as np
import math

p={}
p['g'] = 9.8;# % Gravity constant


#% Body mass* (used in simulation, estimated from real robot) = 2.2337 kg
p['mb'] = 4.4;# %2.2337; % kg

#% Body inertia* = diag([0.0030709, 0.0046624, 0.0035462])
#% IB = [X Y Z]
p['Ib_1'] = 0.0030709;
p['Ib_2'] = 0.0046624;
p['Ib_3'] = 0.0035462;

#% COM coordinates* (x,y,z) relative to body center


p['d_body_com'] = np.array([[0.021107], [0], [-0.041523]]) #% Husky robot actual COM

#% hip positions (body frame)
p['dHip'] = np.zeros((3,4));
p['dHip'][:,0] = np.array([[0.19745], [-0.045], [0]]).T  - p['d_body_com'].T
p['dHip'][:,1] = np.array([ [0.19745],  [0.045], [0]]).T - p['d_body_com'].T
p['dHip'][:,2] = np.array([[-0.19745], [-0.045], [0]]).T - p['d_body_com'].T
p['dHip'][:,3] = np.array([[-0.19745],  [0.045], [0]]).T - p['d_body_com'].T

p['dHip']=p['dHip'] 
#% Ground position in inertial frame
p['ground_z'] = 0;

#%% Ground and compliant model Parameters

#% Compliant ground model ==================================================
p['kp_g'] = 8000;# % 400; %  Spring
p['kd_g'] = np.sqrt(p['kp_g'])*2*1.4; #% 200 Damping

#% Ground friction model ===================================================

#% Basic friction model parameters
p['kfc'] = 0.54;   # % Coulomb friction coefficient
p['kfs'] = 0.60;  # % Static friction coefficient
p['kfb'] = 0.85;   #% Viscous friction coefficient
p['vs'] = 1e-2;     #   % Stribeck velocity (m/s)

#% LuGre model parameters
p['s0'] = 100;       #          % LuGre bristle stiffness
p['s1'] = np.sqrt(p['s0'])*2*0.5; #   % LuGre bristle damping
p['gv_min'] = 1e-3;    #% static/coulomb friction threshold for no contact assumption
p['zd_max'] = 100;    #% z decay rate during no contact
p['vs'] = 1e-3;        #% Stribeck velocity (m/s)

#% Compliant leg parameters ================================================

#% Thin tube 2nd moment of area
p['R1'] = 0.005; #% Leg beam inner radius (m), assuming a thin cylindrical rod
p['R2'] = 0.004; #% Leg beam outer radius (m), assuming a thin cylindrical rod
p['I'] = (np.power(p['R2'],4) - np.power(p['R1'],4))*math.pi/2;   

#% p.E = 95e9;   % Average carbon fiber modulus of elasticity (Pa)
p['E'] = 1e9;   #% Modulus of elasticity (Pa)

#% Set to zero to transform this model into HROM rigid
p['kc_p'] = 1000*1*0;  #% Spring constant of the deflection model
p['kc_d'] = np.sqrt(p['kc_p'])*2*1.4;  # % Damping constant of the deflection model


#%% Controller and observer parameters

#% Observer model parameters ===============================================

p['K_mbdo'] = np.diag([1, 1, 1, 1, 1, 1])*200;  #  % Observer gain

#% Joint controller ========================================================

#% Joint PD controller gains
p['Kd'] = 2500;            #% Proportional gain
p['Kp'] = (np.power(p['Kd']/2,2));      #% Derivative gain

#% Touchdown estimation ====================================================

#% Probabilistic model, phase based, contact. [FR, FL, HR, HL]
p['mean_ph_c0'] = np.array([[0.5],[0],[0],[0.5]]);   #% phase at the beginning of contact
p['mean_ph_c1'] = np.array([[1],[0.5],[0.5],[1]]);   #% phase at the end of contact
p['std_ph_c0'] = 0.04;             #% Standard deviation
p['std_ph_c1'] = 0.04;

#% Probabilistic model, phase based, no contact (swing)
p['mean_ph_nc0'] = np.array([[0],[0.5],[0.5],[0]]); #   % phase at the beginning of swing
p['mean_ph_nc1'] = np.array([[0.5],[1],[1],[0.5]]);    #% phase at the end of swing
p['std_ph_nc0'] = 0.04;              #% Standard deviation
p['std_ph_nc1'] = 0.04;

#% Probabilistic model, gcf based (z direction only)
p['mean_gcf'] = np.ones((4,1))*10 #   % About 10 N for all legs when idling
p['std_gcf'] = 1;

#% Probabilistic model, foot position based (z direction only)
p['mean_pz'] = np.array([[5],[5],[5],[5]]);
p['std_pz'] = 1;    



# Debug settings
p['debug_skip_animation'] = 1;
p['debug_optimization'] = 0;
p['debug_save_results'] = 0;

# Constraint:
# 1: roll and pitch
# else: no constraint
p['use_constraint'] = 1;

# Model parameters (stored in p)
p['use_lugre'] = 0;            # 1 = enable lugre friction model
p['use_damped_rebound'] = 0;   # 1 = enable damped rebound, very plastic impact
p['use_centered_mass'] = 0;    # 0 = use Husky front heavy mass distribution
      

# Simulation settings
p['dt'] = 0.0001;       # Simulation discrete time step (s)
p['t_end'] = 5;       # Simulation discrete time step (s)
p['record_data'] = 1;    # Record simulation data in 'Data' struct
p['print_progress'] = 1; # Display simulation progress

# Gait parameters (NOTE: must be fully divisible by p.dt)
p['gait_start'] = 0.1;         # Time to start walking (s)
p['gait_step_period'] = 0.24;  # Step period (s)
p['gait_wait_period'] = 0.01;  # Wait time after end of swing phase (s)

# Initial states
p['init_leg_length'] = 0.3;    # initial leg length, straight down
p['leg_swing_height'] = 0.1;   # swing phase leg height (bezier coefficient)



# Flags ===================================================================

# Debug settings
p['debug_skip_animation'] = 1;
p['debug_optimization'] = 0;
p['debug_save_results'] = 0;

# Constraint:
# 1: roll and pitch
# else: no constraint
p['use_constraint'] = 1;

# Model parameters (stored in p)
p['use_lugre'] = 0;            # 1 = enable lugre friction model
p['use_damped_rebound'] = 0;   # 1 = enable damped rebound, very plastic impact
p['use_centered_mass'] = 0;    # 0 = use Husky front heavy mass distribution
#load_husky_parameters       % Load the other Husky and physical model parameters

# Simulation settings
p['dt'] = 0.0001;       # Simulation discrete time step (s)
p['t_end'] = 5;       # Simulation discrete time step (s)
p['record_data'] = 1;    # Record simulation data in 'Data' struct
p['print_progress'] = 1; # Display simulation progress

# Gait parameters (NOTE: must be fully divisible by p.dt)
p['gait_start'] = 0.1;         # Time to start walking (s)
p['gait_step_period'] = 0.24;  # Step period (s)
p['gait_wait_period'] = 0.01;  # Wait time after end of swing phase (s)

# Initial states
p['init_leg_length'] = 0.3;    # initial leg length, straight down
p['leg_swing_height'] = 0.1;   # swing phase leg height (bezier coefficient)


#%% Simulation / Optimization

if (p['debug_optimization'] == 1):
    # Perform nonlinear optimization on something
    print('N/A')
  
  
else:
    # Perform standard simulation
    [cost, Data] = hromSimulation(p)
  
  # Save simulation data
  #if (p.['debug_save_results' == 1)
   # save('simulation_data/test.mat', 'Data', 'p');
  #end
#end

#%% Plotting and animation


#if (p.debug_skip_animation == 0)
 # plot_animation_script
#end
#plot_data_script


# params = np.block([p['mb'], p['g'], p['Ib_1'], p['Ib_2'], p['Ib_3'], p['dHip'].reshape(p['dHip'].size), p['ground_z']]);
# params=params.reshape((params.size,1))
# Nx=66
# x=np.ones((Nx,1))

# [M, h, Bg] = func_MhBe(x, params)
# ue = np.zeros((6,1))
# uj=np.zeros((12,1))
# [f1, ug, pos_feet, vel_feet, del_ref]=hrom_compliant_model(x, ue, uj, p)
