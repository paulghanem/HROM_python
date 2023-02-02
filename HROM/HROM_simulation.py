#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:24:06 2023

@author: paul
"""
import autograd.numpy as np
import math 
from functions_HROM import *
from HROM_compliant_model import *



def hromSimulation(p):

    # Initialization ========================================================
    
    cost = 0; # cost for optimization, if any
    Data={}
   
    time=np.linspace(0,p['t_end'],int(p['t_end']/p['dt']))
    N = len(time);
    
    # Gait step lengths
    N_step = p['gait_step_period'] / p['dt'];
    N_wait = p['gait_wait_period'] / p['dt'];
    N_gait = np.array([N_step, N_wait, N_step, N_wait]);
    gait_state = 4; # 1: swing 13, 2: wait, 3: swing 24, 4: wait
    
    # Initialize states
    Nx = 66; # Number of states to record
    x = np.zeros((Nx,1));
    x[8:12] = np.ones((4,1))*p['init_leg_length'];    # initial leg length
    x[14] = p['init_leg_length'] + p['d_body_com'][2]; # initial height
    
    # Rb_init = eye(3);
    Rb_init = rot_z(math.pi*0.9)
    x[15:24] = Rb_init.reshape((Rb_init.size,1),order='F')
    
    
    flag_start = 0; # becomes 1 to start walking
    
    # Controller initialization
    wb_ref = np.zeros((3,1));
    Rb_ref = Rb_init;
    vb_ref = np.zeros((3,1));
    xb_ref = x[12:15];
    
    roll_i = 0;
    pitch_i = 0;
    
    # Initialize data recording
    if p['record_data'] == 1:
      
      Data['t'] = time;
      Data['x'] = np.zeros((Nx,N));           # All simulation states
      Data['ue'] = np.zeros((6,N));           # External forces and torque [f; tau]
      Data['ug'] = np.zeros((12,N));          # Leg ground forces
      Data['uj'] = np.zeros((12,N));          # Virtual leg joint/length acceleration
      Data['rpy'] = np.zeros((3,N));          # Roll / pitch / yaw
      Data['xf'] = np.zeros((12,N));          # Foot inertial positions, order: [FR, FL, RR, FL]
      Data['vf'] = np.zeros((12,N));          # Foot inertial velocities, order: [FR, FL, RR, FL]
      Data['xfB'] = np.zeros((12,N));         # Body frame foot
      Data['del_ref'] = np.zeros((8,N));      # ????
      
      # State machine
      Data['gait_state'] = np.zeros((1,N));
      Data['xb_ref'] = np.zeros((3,N));
      
      # Unused?
    #   Data.MBDO = zeros(6,N);         
    #   Data.f_sum = zeros(3,N);
    #   Data.t_sum = zeros(3,N);
    #   
    #   Data.xfB_t = zeros(12,N);       % ??
      
    
    else:
      Data = {};
    
    
    # Simulation starts =======================================================
    
    if (p['print_progress'] == 1):
      print('Simulation begins!')
    
    
    # Start simulation
    for i in range (N):
      
      # Measurements -------------------------------------------------------
      
      xb = x[12:15];                  # Body inertial position
      vb = x[36:39];                  # Body inertial velocity
      Rb = np.reshape(x[15:24], [3,3],order='F');  # Body rotation matrix
      wb = x[39:42];                  # Body angular velocity (about body frame)
      q = x[0:12];                    # Joint angle and leg length
      qd = x[24:36];                  # Joint ang vel and leg length rate
      
    #   i
      
      # Extract roll pitch and yaw
      
      
    #   P = [1,0,0; 0,0,1; 0,1,0];
    #   R_test = P'*Rb*P;
    #   roll = -atan2(R_test(3,2),R_test(3,3));
    #   pitch = atan2(R_test(2,1),R_test(1,1));
    #   yaw = -real(asin(R_test(3,1)));
    
      yaw = -np.arctan2( Rb[1,0], Rb[0,0] );
      pitch = -np.arctan2( -Rb[2,0], np.sqrt(np.power(Rb[2,1],2) + np.power(Rb[2,2],2)) );
      roll = -np.arctan2( Rb[2,1], Rb[2,2] );
      
      
      # Trajectory --------------------------------------------------------
      
      u_trajectory = np.array([[0.2],[0.3]]) # forward velocity and heading rate
        
      # State machine -----------------------------------------------------
      
      # Initialize state machine
      if ( time[i] >= p['gait_start'] and flag_start == 0):
        flag_start = 1;
        i_start = i;
    
        # Initialize reference states
        xb_ref = x[12:15];
      
      
      # State machine
      if (flag_start == 1):
        
        # Update state machine after some time
        if ( math.remainder(i - i_start, sum(N_gait[0:gait_state]) ) == 0):
          if gait_state == 4:
            i_start = i
            gait_state = 1
          else:
            gait_state = gait_state + 1;
          
          
          # Inertial foot trajectory
          pf_bezier = updateGaitState(u_trajectory, gait_state, pos_feet, xb, yaw, p);
          i_gait_start = i;
    
        
        # Bezier update for foot trajectory
        s = (i - i_gait_start) / N_gait[gait_state-1];
        s = saturate(s, 0, 1);
        
        # Determine the joint/leg length refrences
    #     if gait_state == 1 || gait_state == 3
    #       wb_ref(3) = u_trajectory(2);
    #       Rb_ref = Rb_ref + Rb_ref*skew(wb_ref)*p.dt;
    #       vb_ref = rot_z(yaw) * [u_trajectory(1);0;0];
    #       xb_ref = xb_ref + vb_ref*p.dt;
    #     end
        
        if 1: 
          pos_feetR = np.reshape(pos_feet, [3,4],order='F');
          wb_ref[2] = u_trajectory[1];
          vb_ref = np.matmul(rot_z(yaw) , np.array([[u_trajectory[0]],[0],[0]],dtype=np.float32))
          
          
    #       temp1 = pos_feetR(:,1) - pos_feetR(:,3);
    #       temp2 = pos_feetR(:,2) - pos_feetR(:,4);
    #       foot_orientation_1 = atan2(temp1(2), temp1(1));
    #       foot_orientation_2 = atan2(temp2(2), temp2(1));
    #       mean_heading = (foot_orientation_1 + foot_orientation_2)/2;
          
          temp1 = np.mean(pos_feetR[:,[0,1]], axis=1);
          temp1=temp1.reshape((3,1),order='F')
          temp2 = np.mean(pos_feetR[:,[2,3]], axis=1);
          temp2=temp2.reshape((3,1),order='F')
          temp3 = temp1 - temp2;
          mean_heading = np.arctan2(temp3[1], temp3[0]);      
          Rb_ref = rot_z(mean_heading);
          
          temp = np.mean(pos_feetR.T,axis=0);
          temp=temp.reshape((3,1),order='F')
          
          temp2 = np.matmul(Rb_ref,(p['d_body_com']));
          xb_ref[0:2] = temp[0:2] + temp2[0:2];
        
    
        [pf_ref, vf_ref] = footBezier(s, pf_bezier);
        
        # Orientation adjustments
        roll_i = roll_i + roll * p['dt'];
        pitch_i = pitch_i + pitch * p['dt'];
        d_roll = -roll * 0. - roll_i * 0.0;
        d_pitch = -pitch * 0. - pitch_i * 0.;
        
    #     pf_ref(3,:) = pf_ref(3,:) + [d_roll, -d_roll, d_roll, -d_roll];
    #     pf_ref(3,:) = pf_ref(3,:) + [d_pitch, d_pitch, -d_pitch, -d_pitch];
        
        [q_ref, qd_ref] = jointReference(pf_ref, vf_ref, xb_ref, Rb_ref, vb_ref, wb_ref, p);
    #     [q_ref, qd_ref] = jointReference(pf_ref, vf_ref, xb, Rb, vb, wb, p);
    
        # Extend the legs
    #     q_ref(9:12) = q_ref(9:12) - [d_roll, -d_roll, d_roll, -d_roll]';
    #     q_ref(9:12) = q_ref(9:12) - [d_pitch, d_pitch, -d_pitch, -d_pitch]';
    
        # DEBUG
    #     temp = 6;
    #     q_ref(5) = q_ref(5) + p.dt*temp;
    #     qd_ref(5) = temp;
    
      else:
        # No active control
        q_ref = x[0:12];
        qd_ref = x[24:36];
     
      
      # Joint / leg controller -------------------------------------------
      
      uj = p['Kp']*(q_ref - q) + p['Kd']*(qd_ref - qd)
      
      # External forces -------------------------------------------------
      
      ue = np.zeros((6,1));
      
      # Advance in time -------------------------------------------------
        
      # March time step using RK4
      [xk, ug, pos_feet, vel_feet, del_ref] = march_rk4(x, ue, uj, p);
      
      # Record data
      if p['record_data'] == 1:
    
        # Update the data struct
        Data['x'][:,i] = x.flatten();
        Data['ue'][:,i] = ue.flatten();
        Data['ug'][:,i] = ug.flatten();
        Data['uj'][:,i] = uj.flatten();
        Data['rpy'][:,i] = np.array([[roll],[pitch],[yaw]]).flatten();
        Data['xf'][:,i] = pos_feet.flatten();
        Data['vf'][:,i] = vel_feet.flatten();
        Data['del_ref'][:,i] = del_ref.flatten();
    
        # foot position in body frame
        Data['xfB'][0:3,i] = np.matmul(Rb.T,(pos_feet[0:3] -  x[12:15])).flatten();
        Data['xfB'][3:6,i] = np.matmul(Rb.T,(pos_feet[3:6] -  x[12:15])).flatten();
        Data['xfB'][6:9,i] = np.matmul(Rb.T,(pos_feet[6:9] -  x[12:15])).flatten();
        Data['xfB'][9:12,i] = np.matmul(Rb.T,(pos_feet[9:12] -  x[12:15])).flatten();
    
        #Data['q_des'][:,i] = q_ref.flatten();
        Data['xb_ref'][:,i] = xb_ref.flatten();
        
        # State machine
        Data['gait_state'][:,i] = gait_state;
      
      
      x = xk;     # update next step
      
      # Print progress
      if (p['print_progress'] == 1):
        if (math.remainder(i, round(N/10)) == 0):
          print(['Progress: ', str(round(i/N*100)), '%'])
        
     
      
    
    
    if (p['print_progress'] == 1):
      print('Done!')
    
    
    return cost, Data 

#%% Local Functions

# RK4 integration scheme
def march_rk4(x, ue, uj, p):
    [f1, ug, pos_feet, vel_feet, del_ref] = hrom_compliant_model(x, ue, uj, p);
    [f2,_,_,_,_] = hrom_compliant_model(x + f1*p['dt']/2, ue, uj, p);
    [f3,_,_,_,_] = hrom_compliant_model(x + f2*p['dt']/2, ue, uj, p);
    [f4,_,_,_,_] = hrom_compliant_model(x + f3*p['dt'], ue, uj, p);
    xk = x + (f1/6 + f2/3 + f3/3 + f4/6)*p['dt'];
    return xk, ug, pos_feet, vel_feet, del_ref

# Saturation function
def saturate(x, xmin, xmax):
    if (x > xmax):
      y = xmax;
    elif (x < xmin):
      y = xmin;
    else:
      y = x;
    
    return y

# State machine, update bezier coefficients depending on the trajectory
def updateGaitState(u_trajectory, gait_state, pos_feet_raw, xb, yaw, p):

    n_bezier = 6;
    
    b = np.zeros((3,4, n_bezier+1));
    pos_feet = np.reshape(pos_feet_raw, [3,4],order='F');
    
    # Input components
    v_forward = u_trajectory[0];
    heading_rate = u_trajectory[1];
    
    # Initialize bezier coefficients for stationary trajectory
    # Assume all legs are on the ground
    for k in range(n_bezier+1):
      b[:,:,k] = pos_feet;
    
    
    #Linear velocity update
    d_vel = np.zeros((3,4));
    for i in range (4):
      d_vel[:,i] = np.matmul(rot_z(-yaw),np.array([[v_forward],[0],[0]])/2).flatten()
    
    
    # Heading rate update
    d_rot = np.zeros((3,4));
    for i in range(4):
      d_rot[:,i] = np.cross( np.array([[0],[0],[heading_rate]],dtype=np.float32), pos_feet[:,i].reshape((3,1),order='F') - xb ,axis=0).flatten()/2;
    
    
    # Leg swing height 
    d_height = np.zeros((3,4));
    d_height[2,:] = np.ones((1,4))*p['leg_swing_height'];
    
    # Update values depending on the state machine
    if (gait_state == 1):
      swing_leg_id = np.array([1,4]); # Swing 14 (FR, HL)
    elif (gait_state == 3):
      swing_leg_id = np.array([2,3]); # Swing 23 (FL, HR)
    else:
      swing_leg_id = np.array([]);  # wait
    
    
    # Update bezier coefficients for walking
    # for i_leg = swing_leg_id
    #   b(:,i_leg,3) = b(:,i_leg,3) + (d_vel(:,i_leg) + d_rot(:,i_leg))/2 + d_height(:,i_leg);
    #   b(:,i_leg,4) = b(:,i_leg,4) + d_vel(:,i_leg) + d_rot(:,i_leg);
    #   b(:,i_leg,5) = b(:,i_leg,4);
    # end
    
    for i_leg in ( swing_leg_id-1):
      b[:,i_leg,3] = b[:,i_leg,3] + (d_vel[:,i_leg] + d_rot[:,i_leg])/2 + d_height[:,i_leg];
      b[:,i_leg,4] = b[:,i_leg,4] + d_vel[:,i_leg] + d_rot[:,i_leg];
      b[:,i_leg,5] = b[:,i_leg,4];
      b[:,i_leg,6] = b[:,i_leg,4];
    
    
    
    return b

def footBezier(s, b):
    # Calculate foot trajectory and velocity using 4th order Bezier polynomial
    # s = Gait parameter. value between [0,1]. 
    # b = (3,4,5) array of bezier parameters for all foot. 
    
    n = 6; # Bezier polynomial order.
    
    # Position
    pf_ref = np.zeros((3,4));
    for i in range(n+1):
      pf_ref = pf_ref + binomial(n,i) * np.power((1-s),(n-i)) * np.power(s,i) * b[:,:,i];
   
    
    # Velocity
    vf_ref  = np.zeros((3,4))
    for i in range(n):
      vf_ref = vf_ref + n*binomial(n-1,i) * np.power((1-s),(n-1-i)) * np.power(s,i) * (b[:,:,i+1] - b[:,:,i]);
    

    return pf_ref, vf_ref

def binomial(n,k):
    y = math.factorial(n) / math.factorial(k) / math.factorial(n-k);
    return y
    
    
def jointReference(pf, vf, xb, Rb, vb, wb, p):
    # Calculate joint/leg length references from the inertial leg position and
    # velocities using inverse kinematics.
    
    q_ref = np.zeros((4,3))
    qd_ref = np.zeros((4,3))
    
    for i_leg in range (4):
      # Solve body frame foot position and velocities
      xf_b = np.matmul(Rb.T,pf[:,i_leg].reshape((3,1),order='F') - xb)
      dxf_b = np.matmul(Rb.T,vf[:,i_leg].reshape((3,1),order='F') - vb) - np.cross(wb ,xf_b,axis=0);
        
      [qr, qdr] = ikBody(xf_b, dxf_b, p['dHip'][:,i_leg].reshape((3,1),order='F'));
      q_ref[i_leg, :] = qr.T;
      qd_ref[i_leg, :] = qdr.T;
   
    
    q_ref = q_ref.reshape((q_ref.size,1),order='F')
    qd_ref = qd_ref.reshape((qd_ref.size,1),order='F')
    
    return q_ref, qd_ref


def ikBody(xf_b, dxf_b, Lh_b):
    # Inverse Kinematics, only considers one leg at a time. body frame positions
    # inputs: xf_b = desired foot position,
    #         dxf_b = desired foot velocity
    #         Lh_b = length from center of mass to the hip joint
    # outputs: [q,dq] = joint states and velocities
    
    # Prismatic Joint Length
    lp = np.linalg.norm(xf_b - Lh_b);
    y = (xf_b - Lh_b)/lp;
    theta_f = np.arctan2(y[1],-y[2])
    theta_s = np.arctan2(-y[0],np.linalg.norm([y[1],y[2]]))
    q = np.array([[theta_f],[theta_s],[lp]])
    
    # jacobian(xf_dot,dq)
    delK = np.array([[0      ,                           -lp*math.cos(theta_s)          ,      -math.sin(theta_s)],\
      [lp*math.cos(theta_f)*math.cos(theta_s) ,   -lp*math.sin(theta_f)*math.sin(theta_s)  ,  math.sin(theta_f)*math.cos(theta_s)],\
      [lp*math.sin(theta_f)*math.cos(theta_s)   ,  lp*math.cos(theta_f)*math.sin(theta_s)  , -math.cos(theta_f)*math.cos(theta_s)]]);
    dq = np.matmul(delK,dxf_b);
    return q,dq









