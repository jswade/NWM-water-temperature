
# -*- coding: utf-8 -*-
"""

Visualize results of NWM water temperature model calibration

Scripts produce base plots, further editing performed in Adobe Illustrator

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns


## Load observed STs
# Read saved model time in PST
mod_time_pst = pd.read_csv('../NWM-water-temperature/model_calibration/mod_time_pst.csv')

# Convert to series
mod_time_pst = pd.Series(mod_time_pst.iloc[:,0])

# Read HJA observed data 
hja_obs_st = pd.read_csv("../NWM-water-temperature/data_formatting/site_data/hja_st_2019.csv")

# Filter data to gages of interest
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

# Remove first 48 hours of observed record to account for spinup
st_mack = st_mack.iloc[48:,:]
st_look = st_look.iloc[48:,:]

# Set datetime format for hja gages
st_look.DATE_TIME = pd.to_datetime(st_look.DATE_TIME)
st_mack.DATE_TIME = pd.to_datetime(st_mack.DATE_TIME)
        
# Store observed temperatures as arrays
st_mack_obs = st_mack.WATERTEMP_MEAN
st_look_obs = st_look.WATERTEMP_MEAN
mod_time_pst_clip = mod_time_pst[48:]


# Read model runs from CSV
mc_df_M1 = pd.read_csv('../NWM-water-temperature/model_calibration/M1_calibration.csv')        
M1_st_Fa = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M1_st_Fa.csv')        
M1_st_Ma = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M1_st_Ma.csv') 

mc_df_M2 = pd.read_csv('../NWM-water-temperature/model_calibration/M2_calibration.csv')        
M2_st_Fa = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M2_st_Fa.csv')        
M2_st_Ma = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M2_st_Ma.csv') 

mc_df_M3 = pd.read_csv('../NWM-water-temperature/model_calibration/M3_calibration.csv')        
M3_st_Fa = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M3_st_Fa.csv')        
M3_st_Ma = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M3_st_Ma.csv') 

mc_df_M4 = pd.read_csv('../NWM-water-temperature/model_calibration/M4_calibration.csv')        
M4_st_Fa = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M4_st_Fa.csv')        
M4_st_Ma = pd.read_csv('../NWM-water-temperature/model_calibration/st_runs/M4_st_Ma.csv') 

# Remove first 48 hours from modeled temperatures
M1_st_Fa = M1_st_Fa[48:]
M1_st_Ma = M1_st_Ma[48:]

M2_st_Fa = M2_st_Fa[48:]
M2_st_Ma = M2_st_Ma[48:]

M3_st_Fa = M3_st_Fa[48:]
M3_st_Ma = M3_st_Ma[48:]

M4_st_Fa = M4_st_Fa[48:]
M4_st_Ma = M4_st_Ma[48:]


# Create function to add combined error terms for both gages and sort by weighted error
def error_df(mc_df):

    # Calculate combined error rmse metric
    mc_df['error_sum'] = mc_df.rmse_mack + mc_df.rmse_look
    
    # Add column for weighted error of headwaters and outlet
    # Preference for better fit at outlet
    mc_df['error_weight'] = (1/4)*mc_df.rmse_mack + (3/4)*mc_df.rmse_look

    # Sort data frame by combined weighted error
    mc_df_sort = mc_df.sort_values(by=['error_weight'])
    mc_df_sort = mc_df_sort.reset_index() # Remember old index of mc_df

    # Return sorted dataframe
    return(mc_df_sort)

# Add error terms to dataframes and store sorted dataframes
mc_df_M1_sort = error_df(mc_df_M1)
mc_df_M2_sort = error_df(mc_df_M2)
mc_df_M3_sort = error_df(mc_df_M3)
mc_df_M4_sort = error_df(mc_df_M4)

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

# Retrieve top 50 runs from each model formulation
mc_df_M1_50, M1_st_Fa_50, M1_st_Ma_50 = error_top50(mc_df_M1_sort, M1_st_Fa, M1_st_Ma)
mc_df_M2_50, M2_st_Fa_50, M2_st_Ma_50 = error_top50(mc_df_M2_sort, M2_st_Fa, M2_st_Ma)
mc_df_M3_50, M3_st_Fa_50, M3_st_Ma_50 = error_top50(mc_df_M3_sort, M3_st_Fa, M3_st_Ma)
mc_df_M4_50, M4_st_Fa_50, M4_st_Ma_50 = error_top50(mc_df_M4_sort, M4_st_Fa, M4_st_Ma)

# Calculate 95% confidence intervals at each time step for the top 1% of runs (asumes model is well calibrated under formulation_)
ci_M1_Fa_50 = st.t.interval(0.95, len(M1_st_Fa_50), loc=np.mean(M1_st_Fa_50,axis=1),scale=st.sem(M1_st_Fa_50,axis=1))
ci_M1_Ma_50 = st.t.interval(0.95, len(M1_st_Ma_50), loc=np.mean(M1_st_Ma_50,axis=1),scale=st.sem(M1_st_Ma_50,axis=1))

ci_M2_Fa_50 = st.t.interval(0.95, len(M2_st_Fa_50), loc=np.mean(M2_st_Fa_50,axis=1),scale=st.sem(M2_st_Fa_50,axis=1))
ci_M2_Ma_50 = st.t.interval(0.95, len(M2_st_Ma_50), loc=np.mean(M2_st_Ma_50,axis=1),scale=st.sem(M2_st_Ma_50,axis=1))

ci_M3_Fa_50 = st.t.interval(0.95, len(M3_st_Fa_50), loc=np.mean(M3_st_Fa_50,axis=1),scale=st.sem(M3_st_Fa_50,axis=1))
ci_M3_Ma_50 = st.t.interval(0.95, len(M3_st_Ma_50), loc=np.mean(M3_st_Ma_50,axis=1),scale=st.sem(M3_st_Ma_50,axis=1))

ci_M4_Fa_50 = st.t.interval(0.95, len(M4_st_Fa_50), loc=np.mean(M4_st_Fa_50,axis=1),scale=st.sem(M4_st_Fa_50,axis=1))
ci_M4_Ma_50 = st.t.interval(0.95, len(M4_st_Ma_50), loc=np.mean(M4_st_Ma_50,axis=1),scale=st.sem(M4_st_Ma_50,axis=1))

## ST Envelopes: Headwater
# M1: Headwater
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack.DATE_TIME, st_mack.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack.DATE_TIME,ci_M1_Fa_50[0],ci_M1_Fa_50[1],label="M1 (95th percentile)",alpha=0.75,color='#d76127',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([9,17])
# plt.legend()
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M2: Headwater
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack.DATE_TIME, st_mack.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack.DATE_TIME,ci_M2_Fa_50[0],ci_M2_Fa_50[1],label="M2 (95th percentile)",alpha=0.75,color='#1c4896',zorder=2)
plt.ylabel('Temperature (C)')
plt.legend()
plt.ylim([9,17])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M3: Headwater
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack.DATE_TIME, st_mack.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack.DATE_TIME,ci_M3_Fa_50[0],ci_M3_Fa_50[1],label="M3 (95th percentile)",alpha=0.75,color='#1f8541',zorder=2)
plt.ylabel('Temperature (C)')
plt.legend()
plt.ylim([9,17])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M4: Headwater
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_mack.DATE_TIME, st_mack.WATERTEMP_MEAN,label="Headwater",color="black",lw=1.5,zorder=1)
plt.fill_between(st_mack.DATE_TIME,ci_M4_Fa_50[0],ci_M4_Fa_50[1],label="M4 (95th percentile)",alpha=0.75,color='#94193e',zorder=2)
plt.ylabel('Temperature (C)')
# plt.legend()
plt.ylim([9,17])
ax.set_aspect(1.4)
plt.xticks(rotation=45)


## ST Envelopes: Outlet
# Scale in illustrator to match width of headwater plots
# M1: Outlet
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look.DATE_TIME, st_look.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look.DATE_TIME,ci_M1_Ma_50[0],ci_M1_Ma_50[1],label="M1 (95th percentile)",alpha=0.75,color='#d76127',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,20])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M2: Outlet
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look.DATE_TIME, st_look.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look.DATE_TIME,ci_M2_Ma_50[0],ci_M2_Ma_50[1],label="M2 (95th percentile)",alpha=0.75,color='#1c4896',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,20])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M3: Outlet
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look.DATE_TIME, st_look.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look.DATE_TIME,ci_M3_Ma_50[0],ci_M3_Ma_50[1],label="M3 (95th percentile)",alpha=0.75,color='#1f8541',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,20])
ax.set_aspect(1.4)
plt.xticks(rotation=45)

# M4: Outlet
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(st_look.DATE_TIME, st_look.WATERTEMP_MEAN,label="Outlet",color="black",lw=1.5,zorder=1)
plt.fill_between(st_look.DATE_TIME,ci_M4_Ma_50[0],ci_M4_Ma_50[1],label="M4 (95th percentile)",alpha=0.75,color='#94193e',zorder=2)
plt.ylabel('Temperature (C)')
plt.ylim([10.8,20])
ax.set_aspect(1.4)
plt.xticks(rotation=45)


## JITTER PLOTS OF MODEL VALUES

# Gather all RMSE values for each model version
# Split rmse arrays by top values to be highlighted
M1_rmse_look_top = mc_df_M1_sort.rmse_look[0:50]
M1_rmse_look_bot = mc_df_M1_sort.rmse_look[50:]
M1_rmse_mack_top = mc_df_M1_sort.rmse_mack[0:50]
M1_rmse_mack_bot = mc_df_M1_sort.rmse_mack[50:]

M2_rmse_look_top = mc_df_M2_sort.rmse_look[0:50]
M2_rmse_look_bot = mc_df_M2_sort.rmse_look[50:]
M2_rmse_mack_top = mc_df_M2_sort.rmse_mack[0:50]
M2_rmse_mack_bot = mc_df_M2_sort.rmse_mack[50:]

M3_rmse_look_top = mc_df_M3_sort.rmse_look[0:50]
M3_rmse_look_bot = mc_df_M3_sort.rmse_look[50:]
M3_rmse_mack_top = mc_df_M3_sort.rmse_mack[0:50]
M3_rmse_mack_bot = mc_df_M3_sort.rmse_mack[50:]

M4_rmse_look_top = mc_df_M4_sort.rmse_look[0:50]
M4_rmse_look_bot = mc_df_M4_sort.rmse_look[50:]
M4_rmse_mack_top = mc_df_M4_sort.rmse_mack[0:50]
M4_rmse_mack_bot = mc_df_M4_sort.rmse_mack[50:]

# Create labels for arranging jitter plots
outlet_top = np.repeat('Outlet',len(M1_rmse_look_top))
outlet_bot = np.repeat('Outlet',len(M1_rmse_look_bot))
head_top = np.repeat('Head',len(M1_rmse_look_top))
head_bot = np.repeat('Head',len(M1_rmse_look_bot))

# M1 Jitter Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sns.stripplot(x=M1_rmse_mack_bot,y=head_bot,jitter=0.25, color='grey',alpha=0.05)
sns.stripplot(x=M1_rmse_mack_top,y=head_top,jitter=0.25,color='#d95f0e',alpha=0.75)
sns.stripplot(x=M1_rmse_look_bot,y=outlet_bot,jitter=0.25, color='grey',alpha=0.05)
sns.stripplot(x=M1_rmse_look_top,y=outlet_top,jitter=0.25,color='#d95f0e',alpha=0.75)
ax.set_aspect(.5)
plt.xlim([0,5])
plt.xlabel('RMSE (C)')
plt.gca().xaxis.grid(True)

# M2 Jitter Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sns.stripplot(x=M2_rmse_mack_bot,y=head_bot,jitter=0.25, color='grey',alpha=0.05)
sns.stripplot(x=M2_rmse_mack_top,y=head_top,jitter=0.25,color='#1a4697',alpha=0.75)
sns.stripplot(x=M2_rmse_look_bot,y=outlet_bot,jitter=0.25, color='grey',alpha=0.05)
sns.stripplot(x=M2_rmse_look_top,y=outlet_top,jitter=0.25,color='#1a4697',alpha=0.75)
ax.set_aspect(.5)
plt.xlim([0,5])
plt.xlabel('RMSE (C)')
plt.gca().xaxis.grid(True)

# M3 Jitter Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sns.stripplot(x=M3_rmse_mack_bot,y=head_bot,jitter=0.25, color='grey',alpha=0.05)
sns.stripplot(x=M3_rmse_mack_top,y=head_top,jitter=0.25,color='#17842b',alpha=0.75)
sns.stripplot(x=M3_rmse_look_bot,y=outlet_bot,jitter=0.25, color='grey',alpha=0.05)
sns.stripplot(x=M3_rmse_look_top,y=outlet_top,jitter=0.25,color='#17842b',alpha=0.75)
ax.set_aspect(.5)
plt.xlim([0,5])
plt.xlabel('RMSE (C)')
plt.gca().xaxis.grid(True)

# M4 Jitter Plot
fig = plt.figure()
ax = fig.add_subplot(111)
sns.stripplot(x=M4_rmse_mack_bot,y=head_bot,jitter=0.25, color='grey',alpha=0.05)
sns.stripplot(x=M4_rmse_mack_top,y=head_top,jitter=0.25,color='#93193d',alpha=0.75)
sns.stripplot(x=M4_rmse_look_bot,y=outlet_bot,jitter=0.25, color='grey',alpha=0.05)
sns.stripplot(x=M4_rmse_look_top,y=outlet_top,jitter=0.25,color='#93193d',alpha=0.75)
ax.set_aspect(.5)
plt.xlim([0,5])
plt.xlabel('RMSE (C)')
plt.gca().xaxis.grid(True)


## BOXPLOTS OF TOP 1% ERROR METRICS
# Generate labels for arranging boxplots
M1_lab = np.repeat('M1',len(mc_df_M1_50.rmse_mack))
M2_lab = np.repeat('M2',len(mc_df_M2_50.rmse_mack))
M2_lab = np.repeat('M3',len(mc_df_M3_50.rmse_mack))
M4_lab = np.repeat('M4',len(mc_df_M4_50.rmse_mack))

# RMSE Heatwater
# Create dataframe with RMSE headwater columns
rmse_head = pd.DataFrame([mc_df_M1_50.rmse_mack,mc_df_M2_50.rmse_mack,mc_df_M3_50.rmse_mack,mc_df_M4_50.rmse_mack]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=rmse_head, labels=["M1","M2","M3","M4"],patch_artist=True)
# ax.set_aspect(2.5)
plt.gca().yaxis.grid(True)
plt.ylim([0,2.5])
plt.ylabel('RMSE (C)')
fig.set_size_inches(2,3)


# RMSE Outlet
# Create dataframe with RMSE headwater columns
rmse_outlet = pd.DataFrame([mc_df_M1_50.rmse_look,mc_df_M2_50.rmse_look,mc_df_M3_50.rmse_look,mc_df_M4_50.rmse_look]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=rmse_outlet, labels=["M1","M2","M3","M4"],patch_artist=True)
plt.gca().yaxis.grid(True)
plt.ylim([0,2.5])
plt.ylabel('RMSE (C)')
fig.set_size_inches(2,3)


# Max Heatwater
# Create dataframe with mac headwater columns
max_head = pd.DataFrame([mc_df_M1_50.max_mack,mc_df_M2_50.max_mack,mc_df_M3_50.max_mack,mc_df_M4_50.max_mack]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=max_head, labels=["M1","M2","M3","M4"],patch_artist=True)
plt.gca().yaxis.grid(True)
plt.ylim([-3,3])
plt.ylabel('DMax (C)')
fig.set_size_inches(2,3)

# Max Outlet
# Create dataframe with min headwater columns
max_outlet = pd.DataFrame([mc_df_M1_50.max_look,mc_df_M2_50.max_look,mc_df_M3_50.max_look,mc_df_M4_50.max_look]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=max_outlet, labels=["M1","M2","M3","M4"],patch_artist=True)
plt.ylim([-3,3])
plt.gca().yaxis.grid(True)
plt.ylabel('DMax (C)')
fig.set_size_inches(2,3)

# Min Heatwater
# Create dataframe with min headwater columns
min_head = pd.DataFrame([mc_df_M1_50.min_mack,mc_df_M2_50.min_mack,mc_df_M3_50.min_mack,mc_df_M4_50.min_mack]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=min_head, labels=["M1","M2","M3","M4"],patch_artist=True)
plt.gca().yaxis.grid(True)
plt.ylim([-3,3])
plt.ylabel('DMin (C)')
fig.set_size_inches(2,3)

# Min Outlet
# Create dataframe with RMSE headwater columns
min_outlet = pd.DataFrame([mc_df_M1_50.min_look,mc_df_M2_50.min_look,mc_df_M3_50.min_look,mc_df_M4_50.min_look]).T

fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(x=min_outlet, labels=["M1","M2","M3","M4"],patch_artist=True)
plt.ylim([-3,3])
plt.gca().yaxis.grid(True)
plt.ylabel('DMin (C)')
fig.set_size_inches(2,3)


## Calculate summary stats of errors
np.mean(mc_df_M1_50.rmse_mack)
np.mean(mc_df_M2_50.rmse_mack)
np.mean(mc_df_M3_50.rmse_mack)
np.mean(mc_df_M4_50.rmse_mack)

np.mean(mc_df_M1_50.rmse_look)
np.mean(mc_df_M2_50.rmse_look)
np.mean(mc_df_M3_50.rmse_look)
np.mean(mc_df_M4_50.rmse_look)

np.mean(mc_df_M1_50.max_mack)
np.mean(mc_df_M2_50.max_mack)
np.mean(mc_df_M3_50.max_mack)
np.mean(mc_df_M4_50.max_mack)

np.mean(mc_df_M1_50.max_look)
np.mean(mc_df_M2_50.max_look)
np.mean(mc_df_M3_50.max_look)
np.mean(mc_df_M4_50.max_look)

np.mean(mc_df_M1_50.min_mack)
np.mean(mc_df_M2_50.min_mack)
np.mean(mc_df_M3_50.min_mack)
np.mean(mc_df_M4_50.min_mack)

np.mean(mc_df_M1_50.min_look)
np.mean(mc_df_M2_50.min_look)
np.mean(mc_df_M3_50.min_look)
np.mean(mc_df_M4_50.min_look)

# Calculate mean values of each Monte Carlo parameter
np.mean(mc_df_M1_50, axis=0)
np.mean(mc_df_M2_50, axis=0)
np.mean(mc_df_M3_50, axis=0)
np.mean(mc_df_M4_50, axis=0)





