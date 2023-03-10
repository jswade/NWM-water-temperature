#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for model calibration runs for NWM water temperature model
Calibration performed for 4 model configurations using Monte Carlo parameter sampling (5000 runs for each configuration)


Model is run in the HJ Andrews catchment during July 2019

To ensure tributary connections the model is manually run in the following order:
(Fa, Fb, Fc, Fd) -> (Sa) -> (Ma)

Calibration performed in reference to two temperature gages in the basin:
    Lookout Creek (referred to as 'Outlet') - on Ma reach
    Mack Creek (referred to as 'Headwater') - on Fa reach

"""

# Import libraries
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
    
# Import nwm_st_model_base function
# Change working directory to that of the base model script
os.chdir('../NWM-water-temperature/water_temp_model')
from nwm_st_model_base import nwm_st

# Change file back to main folder
os.chdir('..')

# Set node spacing (m) and time step (minutes)
node_space = 1000
time_step = 60

# Define variable ranges to be sampled using Monte Carlo calibration
at_gw1 = np.array([0, 1]) # AT damping coefficient of first-order reaches
at_gw2 = np.array([0, 1]) # AT damping coefficient of second-order reaches
at_gw3 = np.array([0, 1]) # AT damping coefficient of third-order reaches
gw1 = np.array([0.5, 2]) # AT damping coefficient of first-order reaches
gw2 = np.array([0.5, 2]) # AT damping coefficient of second-order reaches
gw3 = np.array([0.5, 2]) # AT damping coefficient of third-order reaches
window = np.array([2, 14]) # Duration of AT-GW moving window (days)
rip = np.array([0.5, 2]) # Riparian shading
hyp_lag = np.array([2,24]) # Hyporheic residence time (hours)
hyp_frac1 = np.array([0, 1]) # Hyporheic flow fraction
hyp_frac2 = np.array([0, 1]) # Hyporheic flow fraction
hyp_frac3 = np.array([0, 1]) # Hyporheic flow fraction

# Define number of Monte Carlo sampes
n_mc = 5000

# Random uniform sampling of parameter spaces
at_gw1_samp = np.random.uniform(at_gw1[0],at_gw1[1],n_mc)
at_gw2_samp = np.random.uniform(at_gw2[0],at_gw2[1],n_mc)
at_gw3_samp = np.random.uniform(at_gw3[0],at_gw3[1],n_mc)
gw1_samp = np.random.uniform(gw1[0],gw1[1],n_mc)
gw2_samp = np.random.uniform(gw2[0],gw2[1],n_mc)
gw3_samp = np.random.uniform(gw3[0],gw3[1],n_mc)
window_samp = np.random.randint(window[0], window[1], n_mc).astype(int)
rip_samp = np.random.uniform(rip[0],rip[1],n_mc)
hyp_lag_samp = np.random.randint(hyp_lag[0],hyp_lag[1],n_mc)
hyp_frac1_samp = np.random.uniform(hyp_frac1[0],hyp_frac1[1],n_mc)
hyp_frac2_samp = np.random.uniform(hyp_frac2[0],hyp_frac2[1],n_mc)
hyp_frac3_samp = np.random.uniform(hyp_frac3[0],hyp_frac3[1],n_mc)

# Assemble sample into dataframe for M1, M2, M3, M4
mc_df_M1 = pd.DataFrame(data={'at_gw1':at_gw1_samp, 'at_gw2':at_gw2_samp, 'at_gw3':at_gw3_samp,
                           'window':window_samp,'rip':rip_samp})

mc_df_M2 = pd.DataFrame(data={'at_gw1':at_gw1_samp, 'at_gw2':at_gw2_samp, 'at_gw3':at_gw3_samp,
                              'window':window_samp,'rip':rip_samp,
                              'gw1':gw1_samp, 'gw2':gw2_samp, 'gw3':gw3_samp})

mc_df_M3 = pd.DataFrame(data={'at_gw1':at_gw1_samp, 'at_gw2':at_gw2_samp, 'at_gw3':at_gw3_samp,
                              'window':window_samp,'rip':rip_samp,
                              'hyp_lag':hyp_lag_samp, 'hyp_frac1':hyp_frac1_samp,
                              'hyp_frac2':hyp_frac2_samp, 'hyp_frac3':hyp_frac3_samp})

mc_df_M4 = pd.DataFrame(data={'at_gw1':at_gw1_samp, 'at_gw2':at_gw2_samp, 'at_gw3':at_gw3_samp,
                              'window':window_samp,'rip':rip_samp,
                              'gw1':gw1_samp, 'gw2':gw2_samp, 'gw3':gw3_samp,
                              'hyp_lag':hyp_lag_samp, 'hyp_frac1':hyp_frac1_samp,
                              'hyp_frac2':hyp_frac2_samp, 'hyp_frac3':hyp_frac3_samp})

# Add columns for error metrics
error_cols = pd.DataFrame()
error_cols['rmse_mack'] = np.linspace(0,0,len(mc_df_M1))
error_cols['rmse_look'] = np.linspace(0,0,len(mc_df_M1))
error_cols['min_mack'] = np.linspace(0,0,len(mc_df_M1))
error_cols['min_look'] = np.linspace(0,0,len(mc_df_M1))
error_cols['max_mack'] = np.linspace(0,0,len(mc_df_M1))
error_cols['max_look'] = np.linspace(0,0,len(mc_df_M1))

# Concatenate error columns to Monte Carlo Dataframes
mc_df_M1 = pd.concat([mc_df_M1, error_cols],axis=1)
mc_df_M2 = pd.concat([mc_df_M2, error_cols],axis=1)
mc_df_M3 = pd.concat([mc_df_M3, error_cols],axis=1)
mc_df_M4 = pd.concat([mc_df_M4, error_cols],axis=1)


# Create function to run Monte Carlo model calibration and output model error metrics
def mod_runs(mc_df, mod_type):
        
    # Initialize arrays to store temperatures from runs at observed gages
    st_Ma = np.zeros((744,n_mc))
    st_Fa = np.zeros((744,n_mc))
    
    ## Load observed STs
    # Read saved model time in PST
    mod_time_pst = pd.read_csv('../NWM-water-temperature/model_calibration/mod_time_pst.csv')

    # Convert to series
    mod_time_pst = pd.Series(mod_time_pst.iloc[:,0])

    # Read HJA observed data 
    hja_obs_st = pd.read_csv("../NWM-water-temperature/data_formatting/site_data/hja_st_2019.csv")
    
    # Filter data to observed gages of interest
    st_look = hja_obs_st[hja_obs_st.SITECODE == "GSLOOK"]
    st_mack = hja_obs_st[hja_obs_st.SITECODE == "GSMACK"]
    
    # Filter HJA observed values by model dates
    st_look = st_look[np.in1d(st_look.DATE_TIME, mod_time_pst)]
    st_mack = st_mack[np.in1d(st_mack.DATE_TIME, mod_time_pst)]
    
    # Filter ST_LOOK by WAT010 probe (Campbell Scientific Model 107)
    st_look = st_look[st_look.WATERTEMP_METHOD == "WAT010"]
    
    # Reset index of dataframes
    st_mack = st_mack.reset_index(drop=True)
    st_look = st_look.reset_index(drop=True)
    
    # Remove first 48 hours of observed record to account for model spinup
    st_mack = st_mack.iloc[48:,:]
    st_look = st_look.iloc[48:,:]
    
    # Set datetime format for hja gages
    st_look.DATE_TIME = pd.to_datetime(st_look.DATE_TIME)
    st_mack.DATE_TIME = pd.to_datetime(st_mack.DATE_TIME)
            
    # Store observed temperatures as arrays
    st_mack_obs = st_mack.WATERTEMP_MEAN
    st_look_obs = st_look.WATERTEMP_MEAN
    mod_time_pst_clip = mod_time_pst[48:]
    
    # Resample st_look and st_mack to daily minima and maxima
    st_look.index = st_look.DATE_TIME
    st_mack.index = st_mack.DATE_TIME
    st_look_min = st_look['WATERTEMP_MEAN'].groupby(pd.Grouper(freq='D')).min()
    st_mack_min = st_mack['WATERTEMP_MEAN'].groupby(pd.Grouper(freq='D')).min()
    st_look_max = st_look['WATERTEMP_MEAN'].groupby(pd.Grouper(freq='D')).max()
    st_mack_max = st_mack['WATERTEMP_MEAN'].groupby(pd.Grouper(freq='D')).max()
    
    # Convert mod_time_pst to datetime
    mod_time_pst = pd.to_datetime(mod_time_pst)

    for l in range(0,n_mc):
    
        print(l)
        
        ## Feed loop different monte carlo dataframe based on model configuration
        if mod_type == "M1":

            # Extract monte carlo sample
            mc_sample = mc_df.iloc[l,0:5]
            
        ## Feed loop different monte carlo dataframe based on configuration
        elif mod_type == "M2":

            # Extract monte carlo sample
            mc_sample = mc_df.iloc[l,0:8]
            
        elif mod_type == "M3":
            
            # Extract monte carlo sample
            mc_sample = mc_df_M3.iloc[l,0:12]
            
        elif mod_type == "M4":
            
            # Extract monte carlo sample
            mc_sample = mc_df.iloc[l,0:9]
    
        # Run first model computation cycle: Fa, Fb, Fc, Fd reaches
        T_Fa, Q_Fa, mod_time_Fa, nodes_Fa = nwm_st('Fa',node_space, time_step, mc_sample)
        T_Fb, Q_Fb, mod_time_Fb, nodes_Fb = nwm_st('Fb',node_space, time_step, mc_sample)
        T_Fc, Q_Fc, mod_time_Fc, nodes_Fc = nwm_st('Fc',node_space, time_step, mc_sample)
        T_Fd, Q_Fd, mod_time_Fd, nodes_Fd = nwm_st('Fd',node_space, time_step, mc_sample)
        
        # Combine T and Q at final node of tributaries into single matrix
        Fa_mat = np.vstack([T_Fa[:,-1], Q_Fa[:,-1]]).T
        Fb_mat = np.vstack([T_Fb[:,-1], Q_Fb[:,-1]]).T
        Fc_mat = np.vstack([T_Fc[:,-1], Q_Fc[:,-1]]).T
        Fd_mat = np.vstack([T_Fd[:,-1], Q_Fd[:,-1]]).T
        
        # Run second model computation cycle: Sa reach
        T_Sa, Q_Sa, mod_time_Sa, nodes_Sa = nwm_st('Sa', node_space, time_step, mc_sample, Fc=Fc_mat)
        
        # Combine T and Q at final node of tributaries into single matrix
        Sa_mat = np.vstack([T_Sa[:,-1], Q_Sa[:,-1]]).T
        
        # Run third model computation cycle: Ma reach
        T_Ma, Q_Ma, mod_time_Ma, nodes_Ma = nwm_st('Ma', node_space, time_step, mc_sample, Fa=Fa_mat, Fb=Fb_mat,Fd=Fd_mat,Sa=Sa_mat)
        
        # Identify model temperatures where HJA Lookout Creek gage is located 
        Ma_look = T_Ma[:,np.argmin(np.abs(nodes_Ma-15107))] 
        
        # Identify model temperatures where HJA Mack Creek gage is located
        Fa_mack = T_Fa[:,np.argmin(np.abs(nodes_Fa-3435))]
        
        # Add temperature predictions to arrays for output
        st_Fa[:,l] = Fa_mack
        st_Ma[:,l] = Ma_look
        
        # Calculate error metrics at each gage
        # Check for errors in model (in case of overflow)
        if np.isnan(Ma_look).any(): 
            
            # Store Nan
            mc_df.rmse_mack[l] = float('NaN')
            mc_df.rmse_look[l] = float('NaN')
            
            # Store Nan
            mc_df.min_mack[l] = float('NaN')
            mc_df.min_look[l] = float('NaN')
            
            # Store Nan
            mc_df.max_mack[l] = float('NaN')
            mc_df.max_look[l] = float('NaN')
            
        else:    

            # Remove first 48 hours of predicted record to account for model spinup
            Fa_mack = Fa_mack[48:]
            Ma_look = Ma_look[48:]

            # Calculate mean squared error
            mse_mack = mean_squared_error(st_mack_obs, Fa_mack)
            mse_look = mean_squared_error(st_look_obs, Ma_look)
            
            # Store RMSE
            mc_df.rmse_mack[l] = math.sqrt(mse_mack)
            mc_df.rmse_look[l] = math.sqrt(mse_look)

            # Resampled modeled temperatures to daily minima and maxima
            T_df = pd.DataFrame(data={'Fa_mack':Fa_mack, 'Ma_look':Ma_look})
            T_df.index = pd.to_datetime(mod_time_pst_clip)
            Ma_look_min = T_df['Ma_look'].groupby(pd.Grouper(freq='D')).min()
            Fa_mack_min = T_df['Fa_mack'].groupby(pd.Grouper(freq='D')).min()
            Ma_look_max = T_df['Ma_look'].groupby(pd.Grouper(freq='D')).max()
            Fa_mack_max = T_df['Fa_mack'].groupby(pd.Grouper(freq='D')).max()
            
            # Calculate difference average error from daily minima and maxima
            mc_df.min_mack[l] = np.mean(Fa_mack_min - st_mack_min)
            mc_df.min_look[l] = np.mean(Ma_look_min - st_look_min)
            mc_df.max_mack[l] = np.mean(Fa_mack_max - st_mack_max)
            mc_df.max_look[l] = np.mean(Ma_look_max - st_look_max) 
            
    # Convert temperature arrays to dataframes and add datetime column
    st_Fa = pd.concat([mod_time_pst, pd.DataFrame(st_Fa)], axis=1)  
    st_Ma = pd.concat([mod_time_pst, pd.DataFrame(st_Ma)], axis=1)    
            
    return(mc_df,st_Ma, st_Fa)


# Run Monte Carlo calibration on model configurations (M1, M2, M3, M4)   
# Store error metrics data frame, predicted temperature at both observed gages
# Note: This function doesn't return values when it is stopped before completion
# Depending on runtime on your computer, you may need to divide these runs into smaller batches
M1, M1_st_Ma, M1_st_Fa = mod_runs(mc_df_M1,'M1')  
M2, M2_st_Ma, M2_st_Fa = mod_runs(mc_df_M2,'M2')  
M3, M3_st_Ma, M3_st_Fa = mod_runs(mc_df_M3,'M3')    
M4, M4_st_Ma, M4_st_Fa = mod_runs(mc_df_M4,'M4')         

# Write error metrics and predicted water temperatures at two gages to csv
mc_df_M1.to_csv('../NWM-water-temperature/model_calibration/M1_calibration.csv', index=False)        
M1_st_Fa.to_csv('../NWM-water-temperature/model_calibration/st_runs/M1_st_headwater.csv', index=False)        
M1_st_Ma.to_csv('../NWM-water-temperature/model_calibration/st_runs/M1_st_outlet.csv', index=False) 

M2.to_csv('../NWM-water-temperature/model_calibration/M2_calibration.csv', index=False)        
M2_st_Fa.to_csv('../NWM-water-temperature/model_calibration/st_runs/M2_st_headwater.csv', index=False)        
M2_st_Ma.to_csv('../NWM-water-temperature/model_calibration/st_runs/M2_st_outlet.csv', index=False) 

M3.to_csv('../NWM-water-temperature/model_calibration/M3_calibration.csv', index=False)        
M3_st_Fa.to_csv('../NWM-water-temperature/model_calibration/st_runs/M3_st_headwater.csv', index=False)        
M3_st_Ma.to_csv('../NWM-water-temperature/model_calibration/st_runs/M3_st_outlet.csv', index=False)

M4.to_csv('../NWM-water-temperature/model_calibration/M4_calibration.csv', index=False)        
M4_st_Fa.to_csv('../NWM-water-temperature/model_calibration/st_runs/M4_st_headwater.csv', index=False)        
M4_st_Ma.to_csv('../NWM-water-temperature/model_calibration/st_runs/M1_st_outlet.csv', index=False) 


