#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:24:18 2023

@author: paul
"""

from functions_HROM import *
import autograd.numpy as np
import math



def hrom_compliant_model(x,ue,uj,p):
    
    # x = states = [theta_f; theta_s; l_pris; x_body; R_body(:); ...
    #               thetaD_f; thetaD_s; lD_pris; v_body; w_body]
    # ue = external forces acting on the body, ue = [f_in; tau_body]
    # uj = joint space acceleration = [theta_f_DD; theta_s_DD; l_pris_DD]
    # t = current time
    # p = parameter struct
    
    # Get the states, if you need them
    # qf = x(1:4);        % Hip frontal angles [FR, FL, HR, HL] 
    # qs = x(5:8);        % Hip sagittal angles
    lp = x[8:12];       # Leg prismatic joint length
    # xb = x(13:15);      % Body inertial position
    Rb = np.reshape(x[15:24], [3,3],order='F');   # Body rotation matrix
    # qsD = x(25:28);     % Hip frontal ang velocity
    # qsD = x(29:32);     % Hip sagittal ang velocity
    # lpD = x(33:36);     % Leg prismatic joint length rate of change
    # vb = x(37:39);      % Body inertial velocity
    wb = x[39:42];        # Body angular velocity (about body frame)
    
    # df = x(43:50);      % Feet displacements due to compliance
    # dfD = x(51:58);     % Feet displacements rate of change due to compliance
    z = x[58:66];       # Lugre model z (FR_xy, FL_xy, etc)
    
    # Used in calculating the functions generated symbolically
    # params = [p.mb; p.g; p.Ib_1; p.Ib_2; p.Ib_3; ...
    #           p.lh_FR; p.lh_FL; p.lh_HR; p.lh_HL; p.ground_z];
    
    params = np.block([p['mb'], p['g'], p['Ib_1'], p['Ib_2'], p['Ib_3'], p['dHip'].reshape(p['dHip'].size,order='F'), p['ground_z']]);
    params=params.reshape((params.size,1),order='F')
    
    #%% Non-constraints forces ==================================================
    
    # Calculate the dynamic model variables, M*acc + h = Bg*ug
    [M, h, Bg] = func_MhBe(x, params);
    RbD =np.matmul(Rb,skew(wb));
    
    # Determine non-constraint ground contact forces (on each legs)
    [pos_feet, vel_feet] = func_feet(x, params);
    
    # Ground reaction forces
    if (p['use_lugre'] == 1):
      # LuGre friction model
      [fg_FR, zd_FR] = ground_force_model_lugre(pos_feet[0:3], vel_feet[0:3], z[0:2], p);
      [fg_FL, zd_FL] = ground_force_model_lugre(pos_feet[3:6], vel_feet[3:6], z[2:4], p);
      [fg_HR, zd_HR] = ground_force_model_lugre(pos_feet[6:9], vel_feet[6:9], z[4:6], p);
      [fg_HL, zd_HL] = ground_force_model_lugre(pos_feet[9:12], vel_feet[9:12], z[6:8], p);
      zd = np.array([[zd_FR], [zd_FL], [zd_HR], [zd_HL]]);
    else:
      # Coulomb and viscous friction model
      fg_FR = ground_force_model(pos_feet[0:3], vel_feet[0:3], p);
      fg_FL = ground_force_model(pos_feet[3:6], vel_feet[3:6], p);
      fg_HR = ground_force_model(pos_feet[6:9], vel_feet[6:9], p);
      fg_HL = ground_force_model(pos_feet[9:12], vel_feet[9:12], p);
      zd = np.zeros((8,1))
   
    ug = np.block([[fg_FR],[fg_FL],[fg_HR],[fg_HL]])
    ug_model = ug
    
    # Gather all non-constraint forces and joint acceleration
    h0 = -h + ue + np.matmul(Bg,ug); # length 6
    
    # Compliance model ========================================================
    
    # Jacobian of leg inertial position to the leg length space (df_x, df_y, lp)
    [Jl_FR, Jl_FL, Jl_HR, Jl_HL] = func_feet_end_Jacobians(x,params);
    
    # Ground forces mapped to the leg displacements
    fg_FR_mapped = np.matmul(Jl_FR,fg_FR);
    fg_FL_mapped = np.matmul(Jl_FL,fg_FL);
    fg_HR_mapped = np.matmul(Jl_HR,fg_HR);
    fg_HL_mapped = np.matmul(Jl_HL,fg_HL);
    
    # Target deflection depending on the lateral forces (xy only)
    del_ref_FR = fg_FR_mapped[0:2] * np.power(lp[0],3) / (3*p['E']*p['I']);
    del_ref_FL = fg_FL_mapped[0:2] * np.power(lp[1],3) / (3*p['E']*p['I']);
    del_ref_HR = fg_HR_mapped[0:2] * np.power(lp[2],3) / (3*p['E']*p['I']);
    del_ref_HL = fg_HL_mapped[0:2] * np.power(lp[3],3) / (3*p['E']*p['I']);
    del_ref = -np.block([[del_ref_FR[0]], [del_ref_FL[0]], [del_ref_HR[0]], [del_ref_HL[0]],\
                [del_ref_FR[1]], [del_ref_FL[1]], [del_ref_HR[1]], [del_ref_HL[1]]])
    
    # Acceleration for the displacement (spring damping model)
    dfDD = p['kc_p']*(del_ref - x[42:50]) - p['kc_d']*x[50:58];
    
    # % Constraint vel y, angvel xz
    # % Js = [0, 1, 0, 0, 0, 0; ...
    # %       0, 0, 0, 1, 0, 0;
    # %       0, 0, 0, 0, 0, 1];
    # % 
    # % Mc = [M, -Js'; Js, zeros(3,3)];
    # % hc = [h0; zeros(3,1)];
    # % temp = Mc\(hc);
    
    # Constraint
    if p['use_constraint'] == 1:
      # Constraint roll and pitch
      Js = np.array([[0, 0, 0, 1, 0, 0], \
            [0, 0, 0, 0, 1, 0]])
      Mc = np.block([[M, -Js.T], [Js, np.zeros((2,2))]])
      hc = np.block([[h0], [np.zeros((2,1))]])
      temp = np.matmul(np.linalg.inv(Mc),hc)
    else:
      temp = np.matmul(np.linalg.inv(M),h0)
    
    
    qdd = temp[0:6];
    
    
    # Calculate dx/dt =========================================================
    
    xd = x*0;
    
    xd[0:15] = x[24:39];  # Velocities
    xd[15:24] = RbD.reshape((RbD.size,1),order='F')  # Rotation matrix rate of change in SO(3)
    xd[24:36] = uj;       # Joint acceleration
    # xd[36:42] = M\(h0);   # Body accelerations (dynamic model)
    xd[36:42] = qdd;   # Body accelerations (dynamic model)
    
    # Compliant model
    xd[42:50] = x[50:58]; # Foot displacement rate of change
    xd[50:58] = dfDD;     # Foot displacement 2nd derivative
    
    # Friction model (LuGre)
    xd[58:66] = zd;

    return xd, ug_model, pos_feet, vel_feet, del_ref

#%% Local Functions

def ground_force_model(x,v,p):
  # Ground force model using basic friction model

  # x = foot inertial position vector
  # v = foot inertial velocity vector
  
  if (x[2] <= p['ground_z']):
    fz = ground_model(x[2], v[2], p);
    fx = friction_model(v[0], fz, p);
    fy = friction_model(v[1], fz, p);
    f = np.array([[fx],[fy],[fz]])
    f=f.reshape((3,1),order='F')
  else:
    f = np.array([[0],[0],[0]])
  
  
  return f


def friction_model(v,N,p):

  # v = velocity along the surface, N = normal force 
  # Coulomb and viscous friction model
  
  fc = p['kfc'] * np.linalg.norm(N) * np.sign(v);
  fs = p['kfs'] * np.linalg.norm(N) * np.sign(v);
#   f = -(p.kfc * norm(N) * sign(v) + p.kfb * v);

  f = -(fc + (fs-fc)*math.exp(-np.power(np.linalg.norm(v/p['vs']),2)) + p['kfb'] * v);
  
  return f

def ground_force_model_lugre(x,v,z,p):

  # LuGre friction model. Solve for friction force and dz/dt
  # x = foot inertial position vector
  # v = foot inertial velocity vector
  # z = vector of average bristle displacement in Dahl and LuGre model
  #       in x and y directions
  
  if (x([2]) <= p['ground_z']):
    fz = ground_model(x[2], v[2], p)
    flag = true
  else:
    fz = 0;
    flag = false;
  
    
  # Calculate zdx
  [fx,zdx] = friction_model_lugre(v[0], z[0], fz, p);
  [fy,zdy] = friction_model_lugre(v[1], z[1], fz, p);
  
  # fx and fy are zero if not contacting the ground
  if (flag == false):
    fx = 0;
    fy = 0;
  
  
  f = np.array([[fx],[fy],[fz]]);
  zd = np.array([[zdx],[zdy]]);
  return f,zd

def friction_model_lugre(v,z,N,p):

  # v = velocity along the surface, N = normal force 
  # z = average bristle displacement and rate (Dahl and LuGre model)
  
  fc = p['kfc'] * np.linalg.norm(N); # Coulomb friction
  fs = p['kfs'] * np.linalg.norm(N); # Static friction  
  
  gv = fc + (fs-fc)*math.exp(-np.linalg.norm(v/p['vs'])^2)
  
  # If gv smaller than certain threshold, assume leg no longer contacting
  # the ground. Rapidly decay the dz/dt in that case.
  if (gv < p['gv_min']):
    # Lose contact or barely any normal force
    zd = - p['zd_max'] * z;         
    f = 0
  else:
    # LuGre model
    zd = v - p['s0'] * np.linalg.norm(v) / gv * z;         # dz/dt
    f = -(p['s0'] * z + p['s1'] * zd + p['kfb'] * v);  # friction force
  
  
  return f,zd



def ground_model(x,v,p):
  # Assume x < ground_z
  
  if (p['use_damped_rebound'] == 1):
    
    # damped rebound model
    f = -p['kp_g'] * (x - p['ground_z']) - p['kd_g'] * v;
  else:
    # undamped rebound model
    if (v < 0):
      f = -p['kp_g'] * (x - p['ground_z']) - p['kd_g'] * v;
    else:
      f = -p['kp_g'] * (x - p['ground_z']);
    
  
    return f 
