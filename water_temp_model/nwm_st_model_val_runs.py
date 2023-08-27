#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for model validation runs for NWM water temperature model
Validation performed for 4 model configurations using Monte Carlo parameter sampling (5000 runs for each configuration)


Model is run in the HJ Andrews catchment during first two weeks of August 2019

To ensure tributary connections the model is manually run in the following order:
(Fa, Fb, Fc, Fd) -> (Sa) -> (Ma)

Validation performed in reference to two temperature gages in the basin:
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

# Define number of Monte Carlo sampes
n_mc = 5000

# Read in calibration datasets for M1, M2, M3, M4
mc_df_M1 = pd.read_csv('../NWM-water-temperature/model_calibration/M1_calibration.csv')
mc_df_M2 = pd.read_csv('../NWM-water-temperature/model_calibration/M2_calibration.csv')
mc_df_M3 = pd.read_csv('../NWM-water-temperature/model_calibration/M3_calibration.csv')
mc_df_M4 = pd.read_csv('../NWM-water-temperature/model_calibration/M4_calibration.csv')

# # Calculate weighted error of headwaters and outlet
# Preference for better fit at outlet
mc_df_M1['error_weight'] = (1/4)*mc_df_M1.rmse_mack + (3/4)*mc_df_M1.rmse_look
mc_df_M2['error_weight'] = (1/4)*mc_df_M2.rmse_mack + (3/4)*mc_df_M2.rmse_look
mc_df_M3['error_weight'] = (1/4)*mc_df_M3.rmse_mack + (3/4)*mc_df_M3.rmse_look
mc_df_M4['error_weight'] = (1/4)*mc_df_M4.rmse_mack + (3/4)*mc_df_M4.rmse_look

# Reset error columns
mc_df_M1.iloc[:,5:12] = 0
mc_df_M2.iloc[:,8:15] = 0
mc_df_M3.iloc[:,9:16] = 0
mc_df_M4.iloc[:,12:19] = 0


# Create function to run Monte Carlo model validation and output model error metrics
def mod_runs(mc_df, mod_type):


    # Initialize arrays to store temperatures from runs at observed gages
    st_Ma = np.zeros((336,n_mc))
    st_Fa = np.zeros((336,n_mc))
    
    # Read saved model time in PST
    mod_time_pst = pd.read_csv('../NWM-water-temperature/model_validation/mod_time_pst_val.csv')

    # Convert to series
    mod_time_pst = pd.Series(mod_time_pst.iloc[:,0])
    
    # Convert time series to pd_timestamp and back to string to match observed ST times
    mod_time_pst = pd.to_datetime(mod_time_pst).apply(lambda x: str(x))

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
            mc_sample = mc_df.iloc[l,1:6]
            
        ## Feed loop different monte carlo dataframe based on configuration
        elif mod_type == "M2":

            # Extract monte carlo sample
            mc_sample = mc_df.iloc[l,1:9]
            
        elif mod_type == "M3":
            
            # Extract monte carlo sample
            mc_sample = mc_df.iloc[l,1:10]
            
        elif mod_type == "M4":
            
            # Extract monte carlo sample
            mc_sample = mc_df.iloc[l,1:13]
    
        # Run first model computation cycle: Fa, Fb, Fc, Fd reaches
        T_Fa, Q_Fa, mod_time_Fa, nodes_Fa = nwm_st('Fa',node_space, time_step, mc_sample,'val')
        T_Fb, Q_Fb, mod_time_Fb, nodes_Fb = nwm_st('Fb',node_space, time_step, mc_sample,'val')
        T_Fc, Q_Fc, mod_time_Fc, nodes_Fc = nwm_st('Fc',node_space, time_step, mc_sample,'val')
        T_Fd, Q_Fd, mod_time_Fd, nodes_Fd = nwm_st('Fd',node_space, time_step, mc_sample,'val')
        
        # Combine T and Q at final node of tributaries into single matrix
        Fa_mat = np.vstack([T_Fa[:,-1], Q_Fa[:,-1]]).T
        Fb_mat = np.vstack([T_Fb[:,-1], Q_Fb[:,-1]]).T
        Fc_mat = np.vstack([T_Fc[:,-1], Q_Fc[:,-1]]).T
        Fd_mat = np.vstack([T_Fd[:,-1], Q_Fd[:,-1]]).T
        
        # Run second model computation cycle: Sa reach
        T_Sa, Q_Sa, mod_time_Sa, nodes_Sa = nwm_st('Sa', node_space, time_step, mc_sample, 'val',Fc=Fc_mat)
        
        # Combine T and Q at final node of tributaries into single matrix
        Sa_mat = np.vstack([T_Sa[:,-1], Q_Sa[:,-1]]).T
        
        # Run third model computation cycle: Ma reach
        T_Ma, Q_Ma, mod_time_Ma, nodes_Ma = nwm_st('Ma', node_space, time_step, mc_sample, 'val',Fa=Fa_mat, Fb=Fb_mat,Fd=Fd_mat,Sa=Sa_mat)
        
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
            
            # Calculate weighted error
            mc_df.error_weight[l] = (1/4)*mc_df.rmse_mack[l] + (3/4)*mc_df.rmse_look[l]
            
    # Convert temperature arrays to dataframes and add datetime column
    st_Fa = pd.concat([mod_time_pst, pd.DataFrame(st_Fa)], axis=1)  
    st_Ma = pd.concat([mod_time_pst, pd.DataFrame(st_Ma)], axis=1)    
            
    return(mc_df,st_Ma, st_Fa)


# Run Monte Carlo validation on model configurations (M1, M2, M3, M4)   
# Store error metrics data frame, predicted temperature at both observed gages
# Note: This function doesn't return values when it is stopped before completion
# Depending on runtime on your computer, you may need to divide these runs into smaller batches
M1, M1_st_Ma, M1_st_Fa = mod_runs(mc_df_M1,'M1')  
M2, M2_st_Ma, M2_st_Fa = mod_runs(mc_df_M2,'M2')  
M3, M3_st_Ma, M3_st_Fa = mod_runs(mc_df_M3,'M3')    
M4, M4_st_Ma, M4_st_Fa = mod_runs(mc_df_M4,'M4')        

# Remove date column from ST files
M1_st_Fa = M1_st_Fa.iloc[:,1:51]
M2_st_Fa = M2_st_Fa.iloc[:,1:51]
M3_st_Fa = M3_st_Fa.iloc[:,1:51]
M4_st_Fa = M4_st_Fa.iloc[:,1:51]

M1_st_Ma = M1_st_Ma.iloc[:,1:51]
M2_st_Ma = M2_st_Ma.iloc[:,1:51]
M3_st_Ma = M3_st_Ma.iloc[:,1:51]
M4_st_Ma = M4_st_Ma.iloc[:,1:51]

# Write error metrics and predicted water temperatures at two gages to csv
M1.to_csv('../NWM-water-temperature/model_validation/M1_validation.csv', index=False)        
M1_st_Fa.to_csv('../NWM-water-temperature/model_validation/st_runs/M1_st_headwater_val.csv', index=False)        
M1_st_Ma.to_csv('../NWM-water-temperature/model_validation/st_runs/M1_st_outlet_val.csv', index=False) 

M2.to_csv('../NWM-water-temperature/model_validation/M2_validation.csv', index=False)        
M2_st_Fa.to_csv('../NWM-water-temperature/model_validation/st_runs/M2_st_headwater_val.csv', index=False)        
M2_st_Ma.to_csv('../NWM-water-temperature/model_validation/st_runs/M2_st_outlet_val.csv', index=False) 

M3.to_csv('../NWM-water-temperature/model_validation/M3_validation.csv', index=False)        
M3_st_Fa.to_csv('../NWM-water-temperature/model_validation/st_runs/M3_st_headwater_val.csv', index=False)        
M3_st_Ma.to_csv('../NWM-water-temperature/model_validation/st_runs/M3_st_outlet_val.csv', index=False)

M4.to_csv('../NWM-water-temperature/model_validation/M4_validation.csv', index=False)        
M4_st_Fa.to_csv('../NWM-water-temperature/model_validation/st_runs/M4_st_headwater_val.csv', index=False)        
M4_st_Ma.to_csv('../NWM-water-temperature/model_validation/st_runs/M4_st_outlet_val.csv', index=False) 


