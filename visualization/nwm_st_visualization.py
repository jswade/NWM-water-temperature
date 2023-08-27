
# -*- coding: utf-8 -*-
"""

Visualize results of NWM water temperature model calibration and validation

Scripts produce base plots, further editing performed in Adobe Illustrator

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns


## Load observed STs
# Read saved model time in PST
mod_time_pst_cal = pd.read_csv('../NWM-water-temperature/model_calibration/mod_time_pst_cal.csv')
mod_time_pst_val = pd.read_csv('../NWM-water-temperature/model_validation/mod_time_pst_val.csv')

# Convert to series
mod_time_pst_cal = pd.Series(mod_time_pst_cal.iloc[:,0])
mod_time_pst_val = pd.Series(mod_time_pst_val.iloc[:,0])

# Convert validation time to datetime and back to string
mod_time_pst_val = pd.to_datetime(mod_time_pst_val)
mod_time_pst_val = mod_time_pst_val.apply(lambda x: str(x))

# Read HJA observed data 
hja_obs_st = pd.read_csv("../NWM-water-temperature/data_formatting/site_data/hja_st_2019.csv")

# Filter data to gages of interest
st_look = hja_obs_st[hja_obs_st.SITECODE == "GSLOOK"]
st_mack = hja_obs_st[hja_obs_st.SITECODE == "GSMACK"]

# Filter ST_LOOK by WAT010 probe (Campbell Scientific Model 107)
st_look = st_look[st_look.WATERTEMP_METHOD == "WAT010"]

# Filter HJA observed values by model dates
st_look_cal = st_look[np.in1d(st_look.DATE_TIME, mod_time_pst_cal)]
st_mack_cal = st_mack[np.in1d(st_mack.DATE_TIME, mod_time_pst_cal)]
st_look_val = st_look[np.in1d(st_look.DATE_TIME, mod_time_pst_val)]
st_mack_val = st_mack[np.in1d(st_mack.DATE_TIME, mod_time_pst_val)]

# Reset index of dataframes
st_mack_cal = st_mack_cal.reset_index(drop=True)
st_look_cal = st_look_cal.reset_index(drop=True)
st_mack_val = st_mack_val.reset_index(drop=True)
st_look_val = st_look_val.reset_index(drop=True)

# Remove first 48 hours of observed record to account for spinup
st_mack_cal = st_mack_cal.iloc[48:,:]
st_look_cal = st_look_cal.iloc[48:,:]
st_mack_val = st_mack_val.iloc[48:,:]
st_look_val = st_look_val.iloc[48:,:]

# Set datetime format for hja gages
st_look_cal.DATE_TIME = pd.to_datetime(st_look_cal.DATE_TIME)
st_mack_cal.DATE_TIME = pd.to_datetime(st_mack_cal.DATE_TIME)
st_look_val.DATE_TIME = pd.to_datetime(st_look_val.DATE_TIME)
st_mack_val.DATE_TIME = pd.to_datetime(st_mack_val.DATE_TIME)
        
# Store observed temperatures as arrays
st_mack_obs_cal = st_mack_cal.WATERTEMP_MEAN
st_look_obs_cal = st_look_cal.WATERTEMP_MEAN
st_mack_obs_val = st_mack_val.WATERTEMP_MEAN
st_look_obs_val = st_look_val.WATERTEMP_MEAN

# Clip time to remove first 48 hours
mod_time_pst_cal_clip = mod_time_pst_cal[48:]
mod_time_pst_val_clip = mod_time_pst_val[48:]

# Read model runs from CSV (cal)
mc_df_M1_cal = pd.read_csv('../NWM-water-temperature/model_calibration/M1_calibration.csv')        
M1_st_Fa_cal = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M1_st_headwater_cal.csv')        
M1_st_Ma_cal = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M1_st_outlet_cal.csv') 

mc_df_M2_cal = pd.read_csv('../NWM-water-temperature/model_calibration/M2_calibration.csv')        
M2_st_Fa_cal = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M2_st_headwater_cal.csv')        
M2_st_Ma_cal = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M2_st_outlet_cal.csv') 

mc_df_M3_cal = pd.read_csv('../NWM-water-temperature/model_calibration/M3_calibration.csv')        
M3_st_Fa_cal = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M3_st_headwater_cal.csv')        
M3_st_Ma_cal = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M3_st_outlet_cal.csv') 

mc_df_M4_cal = pd.read_csv('../NWM-water-temperature/model_calibration/M4_calibration.csv')        
M4_st_Fa_cal = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M4_st_headwater_cal.csv')        
M4_st_Ma_cal = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M4_st_outlet_cal.csv') 

# Read model runs from CSV (val)
mc_df_M1_val = pd.read_csv('../NWM-water-temperature/model_validation/M1_validation_50.csv')        
M1_st_Fa_val = pd.read_csv('../NWM-water-temperature/model_validation/st_runs_50/M1_st_headwater_val.csv')        
M1_st_Ma_val = pd.read_csv('../NWM-water-temperature/model_validation/st_runs_50/M1_st_outlet_val.csv') 

mc_df_M2_val = pd.read_csv('../NWM-water-temperature/model_validation/M2_validation_50.csv')        
M2_st_Fa_val = pd.read_csv('../NWM-water-temperature/model_validation/st_runs_50/M2_st_headwater_val.csv')        
M2_st_Ma_val = pd.read_csv('../NWM-water-temperature/model_validation/st_runs_50/M2_st_outlet_val.csv') 

mc_df_M3_val = pd.read_csv('../NWM-water-temperature/model_validation/M3_validation_50.csv')        
M3_st_Fa_val = pd.read_csv('../NWM-water-temperature/model_validation/st_runs_50/M3_st_headwater_val.csv')        
M3_st_Ma_val = pd.read_csv('../NWM-water-temperature/model_validation/st_runs_50/M3_st_outlet_val.csv') 

mc_df_M4_val = pd.read_csv('../NWM-water-temperature/model_validation/M4_validation_50.csv')        
M4_st_Fa_val = pd.read_csv('../NWM-water-temperature/model_validation/st_runs_50/M4_st_headwater_val.csv')        
M4_st_Ma_val = pd.read_csv('../NWM-water-temperature/model_validation/st_runs_50/M4_st_outlet_val.csv') 


# Remove first 48 hours from modeled temperatures
M1_st_Fa_cal = M1_st_Fa_cal[48:]
M1_st_Ma_cal = M1_st_Ma_cal[48:]

M2_st_Fa_cal = M2_st_Fa_cal[48:]
M2_st_Ma_cal = M2_st_Ma_cal[48:]

M3_st_Fa_cal = M3_st_Fa_cal[48:]
M3_st_Ma_cal = M3_st_Ma_cal[48:]

M4_st_Fa_cal = M4_st_Fa_cal[48:]
M4_st_Ma_cal = M4_st_Ma_cal[48:]

M1_st_Fa_val = M1_st_Fa_val[48:]
M1_st_Ma_val = M1_st_Ma_val[48:]

M2_st_Fa_val = M2_st_Fa_val[48:]
M2_st_Ma_val = M2_st_Ma_val[48:]

M3_st_Fa_val = M3_st_Fa_val[48:]
M3_st_Ma_val = M3_st_Ma_val[48:]

M4_st_Fa_val = M4_st_Fa_val[48:]
M4_st_Ma_val = M4_st_Ma_val[48:]

# Create function to add combined error terms for both gages and sort by weighted error
def error_df(mc_df):

    # Add column for weighted error of headwaters and outlet
    # Preference for better fit at outlet
    mc_df['error_weight'] = (1/4)*mc_df.rmse_mack + (3/4)*mc_df.rmse_look

    # Sort data frame by combined weighted error
    mc_df_sort = mc_df.sort_values(by=['error_weight'])
    mc_df_sort = mc_df_sort.reset_index() # Remember old index of mc_df

    # Return sorted dataframe
    return(mc_df_sort)

# Add error terms to dataframes and store sorted dataframes
mc_df_M1_cal_sort = error_df(mc_df_M1_cal)
mc_df_M2_cal_sort = error_df(mc_df_M2_cal)
mc_df_M3_cal_sort = error_df(mc_df_M3_cal)
mc_df_M4_cal_sort = error_df(mc_df_M4_cal)

# Create function to filter time series to top 50 runs (assumed to represent peak performance of configuration)
def error_top50(mc_df, st_Fa, st_Ma):
    
    # Retrieve index of top 1% of Monte Carlo runs
    top_ind = mc_df['index'][0:50]
    
    # Retrieve mc_df error of 1% of runs
    mc_df_50 = mc_df.iloc[0:50,:]
    
    # Index temperatures at Fa and Ma gages
    st_Fa_50 = st_Fa.iloc[:,top_ind]
    st_Ma_50 = st_Ma.iloc[:,top_ind]
    
    return(mc_df_50, st_Fa_50, st_Ma_50)

    
## WATER TEMPERATURE ENVELOPE PLOTS

# Retrieve top 50 runs from each model formulation (calibration)
mc_df_M1_cal_50, M1_st_Fa_cal_50, M1_st_Ma_cal_50 = error_top50(mc_df_M1_cal_sort, M1_st_Fa_cal, M1_st_Ma_cal)
mc_df_M2_cal_50, M2_st_Fa_cal_50, M2_st_Ma_cal_50 = error_top50(mc_df_M2_cal_sort, M2_st_Fa_cal, M2_st_Ma_cal)
mc_df_M3_cal_50, M3_st_Fa_cal_50, M3_st_Ma_cal_50 = error_top50(mc_df_M3_cal_sort, M3_st_Fa_cal, M3_st_Ma_cal)
mc_df_M4_cal_50, M4_st_Fa_cal_50, M4_st_Ma_cal_50 = error_top50(mc_df_M4_cal_sort, M4_st_Fa_cal, M4_st_Ma_cal)

# Calculate 95% confidence intervals at each time step for the top 1% of runs (asumes model is well calibrated under formulation_)
ci_M1_Fa_cal_50 = st.t.interval(0.95, len(M1_st_Fa_cal_50), loc=np.mean(M1_st_Fa_cal_50,axis=1),scale=st.sem(M1_st_Fa_cal_50,axis=1))
ci_M1_Ma_cal_50 = st.t.interval(0.95, len(M1_st_Ma_cal_50), loc=np.mean(M1_st_Ma_cal_50,axis=1),scale=st.sem(M1_st_Ma_cal_50,axis=1))

ci_M2_Fa_cal_50 = st.t.interval(0.95, len(M2_st_Fa_cal_50), loc=np.mean(M2_st_Fa_cal_50,axis=1),scale=st.sem(M2_st_Fa_cal_50,axis=1))
ci_M2_Ma_cal_50 = st.t.interval(0.95, len(M2_st_Ma_cal_50), loc=np.mean(M2_st_Ma_cal_50,axis=1),scale=st.sem(M2_st_Ma_cal_50,axis=1))

ci_M3_Fa_cal_50 = st.t.interval(0.95, len(M3_st_Fa_cal_50), loc=np.mean(M3_st_Fa_cal_50,axis=1),scale=st.sem(M3_st_Fa_cal_50,axis=1))
ci_M3_Ma_cal_50 = st.t.interval(0.95, len(M3_st_Ma_cal_50), loc=np.mean(M3_st_Ma_cal_50,axis=1),scale=st.sem(M3_st_Ma_cal_50,axis=1))

ci_M4_Fa_cal_50 = st.t.interval(0.95, len(M4_st_Fa_cal_50), loc=np.mean(M4_st_Fa_cal_50,axis=1),scale=st.sem(M4_st_Fa_cal_50,axis=1))
ci_M4_Ma_cal_50 = st.t.interval(0.95, len(M4_st_Ma_cal_50), loc=np.mean(M4_st_Ma_cal_50,axis=1),scale=st.sem(M4_st_Ma_cal_50,axis=1))

ci_M1_Fa_val_50 = st.t.interval(0.95, len(M1_st_Fa_val), loc=np.mean(M1_st_Fa_val,axis=1),scale=st.sem(M1_st_Fa_val,axis=1))
ci_M1_Ma_val_50 = st.t.interval(0.95, len(M1_st_Ma_val), loc=np.mean(M1_st_Ma_val,axis=1),scale=st.sem(M1_st_Ma_val,axis=1))

ci_M2_Fa_val_50 = st.t.interval(0.95, len(M2_st_Fa_val), loc=np.mean(M2_st_Fa_val,axis=1),scale=st.sem(M2_st_Fa_val,axis=1))
ci_M2_Ma_val_50 = st.t.interval(0.95, len(M2_st_Ma_val), loc=np.mean(M2_st_Ma_val,axis=1),scale=st.sem(M2_st_Ma_val,axis=1))

ci_M3_Fa_val_50 = st.t.interval(0.95, len(M3_st_Fa_val), loc=np.mean(M3_st_Fa_val,axis=1),scale=st.sem(M3_st_Fa_val,axis=1))
ci_M3_Ma_val_50 = st.t.interval(0.95, len(M3_st_Ma_val), loc=np.mean(M3_st_Ma_val,axis=1),scale=st.sem(M3_st_Ma_val,axis=1))

ci_M4_Fa_val_50 = st.t.interval(0.95, len(M4_st_Fa_val), loc=np.mean(M4_st_Fa_val,axis=1),scale=st.sem(M4_st_Fa_val,axis=1))
ci_M4_Ma_val_50 = st.t.interval(0.95, len(M4_st_Ma_val), loc=np.mean(M4_st_Ma_val,axis=1),scale=st.sem(M4_st_Ma_val,axis=1))

## ST Envelopes: Headwater (Calibration)
# M1: Headwater (Cal)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack_cal.DATE_TIME, st_mack_cal.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack_cal.DATE_TIME,ci_M1_Fa_cal_50[0],ci_M1_Fa_cal_50[1],label="M1 (95th percentile)",alpha=0.75,color='#d76127',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([9,19])
# plt.legend()
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M2: Headwater (Cal)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack_cal.DATE_TIME, st_mack_cal.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack_cal.DATE_TIME,ci_M2_Fa_cal_50[0],ci_M2_Fa_cal_50[1],label="M2 (95th percentile)",alpha=0.75,color='#1c4896',zorder=2)
plt.ylabel('Temperature (C)')
plt.legend()
plt.ylim([9,19])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M3: Headwater (Cal)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack_cal.DATE_TIME, st_mack_cal.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack_cal.DATE_TIME,ci_M3_Fa_cal_50[0],ci_M3_Fa_cal_50[1],label="M3 (95th percentile)",alpha=0.75,color='#1f8541',zorder=2)
plt.ylabel('Temperature (C)')
plt.legend()
plt.ylim([9,19])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M4: Headwater (Cal)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack_cal.DATE_TIME, st_mack_cal.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack_cal.DATE_TIME,ci_M4_Fa_cal_50[0],ci_M4_Fa_cal_50[1],label="M4 (95th percentile)",alpha=0.75,color='#94193e',zorder=2)
plt.ylabel('Temperature (C)')
# plt.legend()
plt.ylim([9,19])
ax.set_aspect(1.4)
plt.xticks(rotation=45)


## ST Envelopes: Outlet (Calibration)
# Scale in illustrator to match width of headwater plots
# M1: Outlet (Cal)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look_cal.DATE_TIME, st_look_cal.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look_cal.DATE_TIME,ci_M1_Ma_cal_50[0],ci_M1_Ma_cal_50[1],label="M1 (95th percentile)",alpha=0.75,color='#d76127',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,22])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M2: Outlet (Cal)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look_cal.DATE_TIME, st_look_cal.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look_cal.DATE_TIME,ci_M2_Ma_cal_50[0],ci_M2_Ma_cal_50[1],label="M2 (95th percentile)",alpha=0.75,color='#1c4896',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,22])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M3: Outlet (Cal)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look_cal.DATE_TIME, st_look_cal.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look_cal.DATE_TIME,ci_M3_Ma_cal_50[0],ci_M3_Ma_cal_50[1],label="M3 (95th percentile)",alpha=0.75,color='#1f8541',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,22])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M4: Outlet (Cal)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look_cal.DATE_TIME, st_look_cal.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look_cal.DATE_TIME,ci_M4_Ma_cal_50[0],ci_M4_Ma_cal_50[1],label="M4 (95th percentile)",alpha=0.75,color='#94193e',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,22])
ax.set_aspect(1.4)
plt.xticks(rotation=45)


## ST Envelopes: Headwater (Validation)
# M1: Headwater (Val)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack_val.DATE_TIME, st_mack_val.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack_val.DATE_TIME,ci_M1_Fa_val_50[0],ci_M1_Fa_val_50[1],label="M1 (95th percentile)",alpha=0.75,color='#d76127',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([9,19])
# plt.legend()
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M2: Headwater (Val)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack_val.DATE_TIME, st_mack_val.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack_val.DATE_TIME,ci_M2_Fa_val_50[0],ci_M2_Fa_val_50[1],label="M2 (95th percentile)",alpha=0.75,color='#1c4896',zorder=2)
plt.ylabel('Temperature (C)')
# plt.legend()
plt.ylim([9,19])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M3: Headwater (Val)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack_val.DATE_TIME, st_mack_val.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack_val.DATE_TIME,ci_M3_Fa_val_50[0],ci_M3_Fa_val_50[1],label="M3 (95th percentile)",alpha=0.75,color='#1f8541',zorder=2)
plt.ylabel('Temperature (C)')
plt.legend()
plt.ylim([9,19])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M4: Headwater (Val)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack_val.DATE_TIME, st_mack_val.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack_val.DATE_TIME,ci_M4_Fa_val_50[0],ci_M4_Fa_val_50[1],label="M4 (95th percentile)",alpha=0.75,color='#94193e',zorder=2)
plt.ylabel('Temperature (C)')
# plt.legend()
plt.ylim([9,19])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

## ST Envelopes: Outlet (Validation))
# Scale in illustrator to match width of headwater plots
# M1: Outlet (Val)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look_val.DATE_TIME, st_look_val.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look_val.DATE_TIME,ci_M1_Ma_val_50[0],ci_M1_Ma_val_50[1],label="M1 (95th percentile)",alpha=0.75,color='#d76127',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,22])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M2: Outlet (Val)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look_val.DATE_TIME, st_look_val.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look_val.DATE_TIME,ci_M2_Ma_val_50[0],ci_M2_Ma_val_50[1],label="M2 (95th percentile)",alpha=0.75,color='#1c4896',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,22])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M3: Outlet (Val)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look_val.DATE_TIME, st_look_val.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look_val.DATE_TIME,ci_M3_Ma_val_50[0],ci_M3_Ma_val_50[1],label="M3 (95th percentile)",alpha=0.75,color='#1f8541',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,22])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M4: Outlet (Val)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look_val.DATE_TIME, st_look_val.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look_val.DATE_TIME,ci_M4_Ma_val_50[0],ci_M4_Ma_val_50[1],label="M4 (95th percentile)",alpha=0.75,color='#94193e',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,22])
ax.set_aspect(1.4)
plt.xticks(rotation=45)


## JITTER PLOTS OF MODEL VALUES

# Gather all RMSE values for each model version
# Split rmse arrays by top values to be highlighted
M1_rmse_look_cal_top = mc_df_M1_cal_sort.rmse_look[0:50]
M1_rmse_look_cal_bot = mc_df_M1_cal_sort.rmse_look[50:]
M1_rmse_mack_cal_top = mc_df_M1_cal_sort.rmse_mack[0:50]
M1_rmse_mack_cal_bot = mc_df_M1_cal_sort.rmse_mack[50:]

M2_rmse_look_cal_top = mc_df_M2_cal_sort.rmse_look[0:50]
M2_rmse_look_cal_bot = mc_df_M2_cal_sort.rmse_look[50:]
M2_rmse_mack_cal_top = mc_df_M2_cal_sort.rmse_mack[0:50]
M2_rmse_mack_cal_bot = mc_df_M2_cal_sort.rmse_mack[50:]

M3_rmse_look_cal_top = mc_df_M3_cal_sort.rmse_look[0:50]
M3_rmse_look_cal_bot = mc_df_M3_cal_sort.rmse_look[50:]
M3_rmse_mack_cal_top = mc_df_M3_cal_sort.rmse_mack[0:50]
M3_rmse_mack_cal_bot = mc_df_M3_cal_sort.rmse_mack[50:]

M4_rmse_look_cal_top = mc_df_M4_cal_sort.rmse_look[0:50]
M4_rmse_look_cal_bot = mc_df_M4_cal_sort.rmse_look[50:]
M4_rmse_mack_cal_top = mc_df_M4_cal_sort.rmse_mack[0:50]
M4_rmse_mack_cal_bot = mc_df_M4_cal_sort.rmse_mack[50:]

M1_rmse_look_val_top = mc_df_M1_val.rmse_look[0:50]
M1_rmse_mack_val_top = mc_df_M1_val.rmse_mack[0:50]

M2_rmse_look_val_top = mc_df_M2_val.rmse_look[0:50]
M2_rmse_mack_val_top = mc_df_M2_val.rmse_mack[0:50]

M3_rmse_look_val_top = mc_df_M3_val.rmse_look[0:50]
M3_rmse_mack_val_top = mc_df_M3_val.rmse_mack[0:50]

M4_rmse_look_val_top = mc_df_M4_val.rmse_look[0:50]
M4_rmse_mack_val_top = mc_df_M4_val.rmse_mack[0:50]

# Create labels for arranging jitter plots
outlet_top = np.repeat('Outlet (Cal)',len(M1_rmse_look_cal_top))
outlet_bot = np.repeat('Outlet (Cal)',len(M1_rmse_look_cal_bot))
outlet_val = np.repeat('Outlet (Val)',len(M1_rmse_look_val_top))
head_top = np.repeat('Head (Cal)',len(M1_rmse_look_cal_top))
head_bot = np.repeat('Head (Cal)',len(M1_rmse_look_cal_bot))
head_val = np.repeat('Head (Val)',len(M1_rmse_look_val_top))

# M1 Jitter Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sns.stripplot(x=M1_rmse_mack_cal_bot,y=head_bot,jitter=0.15, color='grey',alpha=0.025)
sns.stripplot(x=M1_rmse_mack_cal_top,y=head_top,jitter=0.15,color='#d95f0e',alpha=0.75)
sns.stripplot(x=M1_rmse_mack_val_top,y=head_val,jitter=0.075,color='#d95f0e',marker="^",alpha=0.75)
sns.stripplot(x=M1_rmse_look_cal_bot,y=outlet_bot,jitter=0.15, color='grey',alpha=0.025)
sns.stripplot(x=M1_rmse_look_cal_top,y=outlet_top,jitter=0.15,color='#d95f0e',alpha=0.75)
sns.stripplot(x=M1_rmse_look_val_top,y=outlet_val,jitter=0.075,color='#d95f0e',marker="^",alpha=0.75)
ax.set_aspect(.5)
plt.xlim([0,5])
plt.xlabel('RMSE (C)')
plt.gca().xaxis.grid(True)

# M2 Jitter Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sns.stripplot(x=M2_rmse_mack_cal_bot,y=head_bot,jitter=0.15, color='grey',alpha=0.025)
sns.stripplot(x=M2_rmse_mack_cal_top,y=head_top,jitter=0.15,color='#1a4697',alpha=0.75)
sns.stripplot(x=M2_rmse_mack_val_top,y=head_val,jitter=0.075,color='#1a4697',marker="^",alpha=0.75)
sns.stripplot(x=M2_rmse_look_cal_bot,y=outlet_bot,jitter=0.15, color='grey',alpha=0.025)
sns.stripplot(x=M2_rmse_look_cal_top,y=outlet_top,jitter=0.15,color='#1a4697',alpha=0.75)
sns.stripplot(x=M2_rmse_look_val_top,y=outlet_val,jitter=0.075,color='#1a4697',marker="^",alpha=0.75)
ax.set_aspect(.5)
plt.xlim([0,5])
plt.xlabel('RMSE (C)')
plt.gca().xaxis.grid(True)

# M3 Jitter Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sns.stripplot(x=M3_rmse_mack_cal_bot,y=head_bot,jitter=0.15, color='grey',alpha=0.025)
sns.stripplot(x=M3_rmse_mack_cal_top,y=head_top,jitter=0.15,color='#17842b',alpha=0.75)
sns.stripplot(x=M3_rmse_mack_val_top,y=head_val,jitter=0.075,color='#17842b',marker="^",alpha=0.75)
sns.stripplot(x=M3_rmse_look_cal_bot,y=outlet_bot,jitter=0.15, color='grey',alpha=0.025)
sns.stripplot(x=M3_rmse_look_cal_top,y=outlet_top,jitter=0.15,color='#17842b',alpha=0.75)
sns.stripplot(x=M3_rmse_look_val_top,y=outlet_val,jitter=0.075,color='#17842b',marker="^",alpha=0.75)
ax.set_aspect(.5)
plt.xlim([0,5])
plt.xlabel('RMSE (C)')
plt.gca().xaxis.grid(True)

# M4 Jitter Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sns.stripplot(x=M4_rmse_mack_cal_bot,y=head_bot,jitter=0.15, color='grey',alpha=0.025)
sns.stripplot(x=M4_rmse_mack_cal_top,y=head_top,jitter=0.15,color='#93193d',alpha=0.75)
sns.stripplot(x=M4_rmse_mack_val_top,y=head_val,jitter=0.075,color='#93193d',marker="^",alpha=0.75)
sns.stripplot(x=M4_rmse_look_cal_bot,y=outlet_bot,jitter=0.15, color='grey',alpha=0.025)
sns.stripplot(x=M4_rmse_look_cal_top,y=outlet_top,jitter=0.15,color='#93193d',alpha=0.75)
sns.stripplot(x=M4_rmse_look_val_top,y=outlet_val,jitter=0.075,color='#93193d',marker="^",alpha=0.75)
ax.set_aspect(.5)
plt.xlim([0,5])
plt.xlabel('RMSE (C)')
plt.gca().xaxis.grid(True)


## BOXPLOTS OF TOP 1% ERROR METRICS
# Generate labels for arranging boxplots
M1_lab = np.repeat('M1',len(mc_df_M1_cal_50.rmse_mack))
M2_lab = np.repeat('M2',len(mc_df_M2_cal_50.rmse_mack))
M2_lab = np.repeat('M3',len(mc_df_M3_cal_50.rmse_mack))
M4_lab = np.repeat('M4',len(mc_df_M4_cal_50.rmse_mack))

# RMSE Headwater
# Create dataframe with RMSE headwater columns
rmse_head = pd.DataFrame([mc_df_M1_cal_50.rmse_mack,
                          mc_df_M1_val.rmse_mack,
                          mc_df_M2_cal_50.rmse_mack,
                          mc_df_M2_val.rmse_mack,
                          mc_df_M3_cal_50.rmse_mack,
                          mc_df_M3_val.rmse_mack,
                          mc_df_M4_cal_50.rmse_mack,
                          mc_df_M4_val.rmse_mack,]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=rmse_head, labels=["M1_cal","M1_val","M2_cal","M2_val","M3_cal","M3_val","M4_cal","M4_val"],patch_artist=True)
plt.ylim([0,4])
plt.ylabel('RMSE (C)')
fig.set_size_inches(2,3)


# RMSE Outlet
# Create dataframe with RMSE headwater columns
rmse_outlet = pd.DataFrame([mc_df_M1_cal_50.rmse_look,
                            mc_df_M1_val.rmse_look,
                            mc_df_M2_cal_50.rmse_look,
                            mc_df_M2_val.rmse_look,
                            mc_df_M3_cal_50.rmse_look,
                            mc_df_M3_val.rmse_look,
                            mc_df_M4_cal_50.rmse_look,
                            mc_df_M4_val.rmse_look,]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=rmse_outlet, labels=["M1_cal","M1_val","M2_cal","M2_val","M3_cal","M3_val","M4_cal","M4_val"],patch_artist=True)
plt.gca().yaxis.grid(True)
plt.ylim([0,4])
plt.ylabel('RMSE (C)')
fig.set_size_inches(2,3)


# Max Headwater
# Create dataframe with mac headwater columns
max_head = pd.DataFrame([mc_df_M1_cal_50.max_mack,
                         mc_df_M1_val.max_mack,
                         mc_df_M2_cal_50.max_mack,
                         mc_df_M2_val.max_mack,
                         mc_df_M3_cal_50.max_mack,
                         mc_df_M3_val.max_mack,
                         mc_df_M4_cal_50.max_mack,
                         mc_df_M4_val.max_mack]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=max_head, labels=["M1_cal","M1_val","M2_cal","M2_val","M3_cal","M3_val","M4_cal","M4_val"],patch_artist=True)
plt.gca().yaxis.grid(True)
plt.ylim([-2,6])
plt.ylabel('DMax (C)')
fig.set_size_inches(2,3)

# Max Outlet
# Create dataframe with min headwater columns
max_outlet = pd.DataFrame([mc_df_M1_cal_50.max_look,
                           mc_df_M1_val.max_look,
                           mc_df_M2_cal_50.max_look,
                           mc_df_M2_val.max_look,
                           mc_df_M3_cal_50.max_look,
                           mc_df_M3_val.max_look,
                           mc_df_M4_cal_50.max_look,
                           mc_df_M4_val.max_look]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=max_outlet, labels=["M1_cal","M1_val","M2_cal","M2_val","M3_cal","M3_val","M4_cal","M4_val"],patch_artist=True)
plt.ylim([-2,6])
plt.gca().yaxis.grid(True)
plt.ylabel('DMax (C)')
fig.set_size_inches(2,3)

# Min Heatwater
# Create dataframe with min headwater columns
min_head = pd.DataFrame([mc_df_M1_cal_50.min_mack,
                         mc_df_M1_val.min_mack,
                         mc_df_M2_cal_50.min_mack,
                         mc_df_M2_val.min_mack,
                         mc_df_M3_cal_50.min_mack,
                         mc_df_M3_val.min_mack,
                         mc_df_M4_cal_50.min_mack,
                         mc_df_M4_val.min_mack,]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=min_head, labels=["M1_cal","M1_val","M2_cal","M2_val","M3_cal","M3_val","M4_cal","M4_val"],patch_artist=True)
plt.gca().yaxis.grid(True)
plt.ylim([-4,4])
plt.ylabel('DMin (C)')
fig.set_size_inches(2,3)

# Min Outlet
# Create dataframe with RMSE headwater columns
min_outlet = pd.DataFrame([mc_df_M1_cal_50.min_look,
                           mc_df_M1_val.min_look,
                           mc_df_M2_cal_50.min_look,
                           mc_df_M2_val.min_look,
                           mc_df_M3_cal_50.min_look,
                           mc_df_M3_val.min_look,
                           mc_df_M4_cal_50.min_look,
                           mc_df_M4_val.min_look,]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=min_outlet, labels=["M1_cal","M1_val","M2_cal","M2_val","M3_cal","M3_val","M4_cal","M4_val"],patch_artist=True)
plt.ylim([-4,4])
plt.gca().yaxis.grid(True)
plt.ylabel('DMin (C)')
fig.set_size_inches(2,3)


## Calculate summary stats of errors
# Write function to print summary stats
def stats_out(st):
    
    print("\n" + "Head RMSE = " + str(round(np.mean(st.rmse_mack),2)) + "\n" +
          "Head Max = " + str(round(np.mean(st.max_mack),2)) + "\n" +
          "Head Min = " + str(round(np.mean(st.min_mack),2)) + "\n" +
          "Outlet RMSE = " + str(round(np.mean(st.rmse_look),2)) + "\n" +
          "Outlet Max = " + str(round(np.mean(st.max_look),2)) + "\n" +
          "Outlet Max = " + str(round(np.mean(st.min_look),2)) + "\n")
    
    
stats_out(mc_df_M1_cal_50)
stats_out(mc_df_M2_cal_50)
stats_out(mc_df_M3_cal_50)
stats_out(mc_df_M4_cal_50)

stats_out(mc_df_M1_val)
stats_out(mc_df_M2_val)
stats_out(mc_df_M3_val)
stats_out(mc_df_M4_val)

# Calculate mean values of each Monte Carlo parameter
np.mean(mc_df_M1_cal_50, axis=0)
np.mean(mc_df_M2_cal_50, axis=0)
np.mean(mc_df_M3_cal_50, axis=0)
np.mean(mc_df_M4_cal_50, axis=0)





