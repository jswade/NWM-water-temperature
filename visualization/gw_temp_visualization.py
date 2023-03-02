#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Create plots showing estimated groundwater temperature dependence on coefficients/moving window duration

Groundwater temperatures estimated by lagging and buffering temperatures between mean daily air temperatures and annual air temperatures

Daily mean air temperatures derived from NWM forcing data
Annual air temperatures derived from PRISM data

Credit: PRISM
PRISM Climate Group, Oregon State University, https://prism.oregonstate.edu, data created Dec 2022, accessed 4 Oct 2022.

"""
 
# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz
import matplotlib.dates as mdates
# Ignore shapely warnings
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'


## GROUNDWATER TEMPERATURES AT FA HEADWATER DURING JULY 2019 STUDY PERIOD ##

# Create function to reformat data from input file into dataframe of variable, where rows=time, columns=nodes
# Temporally resamples data from hourly to model time
def data_reformat(df, mod_var, ts):
    
    # df = input data (ns_ma_data)
    # mod_var = variable of interest (string)

    # Get number of reaches in dataframe
    n_reach = len(df.feature_id.unique())
    
    # Index dataframe based on model variable of interest
    df_var = np.array(df[mod_var])
    
    # Reshape dataframe for output  and convert to dataframe (rows=time, columns=nodes)
    df_out = pd.DataFrame(df_var.reshape(int(len(df_var)/n_reach), -1, order='F'))
    
    # Set time indices
    df_out = df_out.set_index(pd.to_datetime(df.nwm_time).unique())
    
    # Create resample object to convert hourly input to model time
    df_resampler = df_out.resample(str(ts)+'T')
    
    # Interpolate data to resampled time interval
    df_out_hourly = df_resampler.interpolate(method='time')
    
    # Convert to numpy array
    out_hourly = df_out_hourly.to_numpy()
    
    # Return formatted matrix
    return(out_hourly)

# Create function to assign model input data to model nodes
# input_val = input value to be interpolated
# input_nodes = Reach distance of input values
# mod_nodes = Reach distance of model nodes
def spat_interp(input_val, input_nodes, mod_nodes):
    
    # Assign values to model nodes based on their centroid location relative to reach segments
    # Remove first 0 value of input nodes to align indices of searchsorted with reach indices
    # Searchsorted index of 0 means node lies within first reach segment
    mod_nodes_loc = np.searchsorted(input_nodes[1:], mod_nodes)

    # Check dimensions of input
    if len(input_val) == n_in: # Array is (n_dt,n)
         
        # Fill in values at model nodes for output based on location of the model nodes
        # Single dimension array
        out_val = input_val[mod_nodes_loc]
    
    elif len(input_val) == n_dt: # Array is (n,)
        
        # Fill in values at model nodes for output based on location of the model nodes
        # Multi-dimension array
        out_val = input_val[:,mod_nodes_loc]
    
    # Convert to array
    out_val = np.array(out_val)
    
    # Return formatted values
    return(out_val)

# Set model domain values
segment = 'Fa' # Mack Creek
node_space = 1000
time_step = 60

## Model Definition ##
    
# Load model forcing and streamflow data (see retrospective_download_hja.py)
hja_data = pd.read_csv('../NWM-water-temperature/data_formatting/nwm_retrospective/retrospective_files/retro_v21_data.csv')

# Load extended LDASIN forcing data for June
ldasin_extend = pd.read_csv('../NWM-water-temperature/data_formatting/nwm_retrospective/retrospective_files/retro_v21_ldasin_extend.csv')

# Load model channel data (see hja_reach_download.py)
hja_df = pd.read_csv('../NWM-water-temperature/data_formatting/nwm_channels/formatted_channels/hja_channel.csv')

# Filter hja_df by reaches 
hja_rch = hja_df[hja_df['segment'] == segment]

# Sort reaches by node number
hja_rch = hja_rch.sort_values(by='hja_id')

# Reset indices of hja_rch
hja_rch = hja_rch.reset_index(drop=True)

# Retrieve list of feature_ids corresponding to selected reach
hja_rch_data = hja_data[hja_data['feature_id'].isin(np.array(hja_rch['feature_id']))]

# Retrieve extended LDASIN data corresponding to reaches
ldasin_rch_data = ldasin_extend[ldasin_extend['feature_id'].isin(np.array(hja_rch['feature_id']))]
    
# Reset ldasin_rch_data index
ldasin_rch_data = ldasin_rch_data.reset_index()

## SPATIAL DOMAIN ##

# Define location of Eulerian model grid
reaches_in = np.zeros(len(hja_rch)+1) # Initialize array of node distances


for p in range(0,len(hja_rch)-1):
    reaches_in[p+1] = hja_rch['Length'][p] + reaches_in[p]
reaches_in[len(hja_rch)] = hja_rch['Length'][len(hja_rch)-1] + reaches_in[len(hja_rch)-1]

# Define number of input reaches
n_in = len(hja_rch)

# Specify distances of segment boundaries (where ST is to be predicted)
nodes = np.arange(0,reaches_in[n_in], node_space)

# Add final node to ensure model nodes are the same length as the input reaches
nodes = np.append(nodes,reaches_in[-1])

# Define number of reach segments (one less than the number of reach boundaries)
# Last model segment will be shorter than node spacing
n = len(nodes)-1

# Calculate actual node length between node boundaries
node_length = np.array([t - s for s, t in zip(nodes, nodes[1:])])

# Define center points of reach segments
# Values of forcing inputs interpolated based on midpoint of reach
nodes_center = nodes[1:] - 0.5 * node_length

# Interpolate stream order to model nodes
stream_order = spat_interp(hja_rch['stream_order'], reaches_in, nodes_center)


## TEMPORAL DOMAIN ##

# Sort input data to find start and end times
hja_rch_data = hja_rch_data.assign(nwm_time = pd.to_datetime(hja_rch_data['nwm_time']))
ldasin_rch_data = ldasin_rch_data.assign(nwm_time = pd.to_datetime(ldasin_rch_data['nwm_time'],format="%Y-%m-%d_%H:%M:%S"))
input_sort = hja_rch_data.sort_values('nwm_time',ignore_index=True)
nwm_time = input_sort['nwm_time']

# Define model start and end time
time_start = np.datetime64(nwm_time[0])
time_end = np.datetime64(nwm_time.iloc[-1]) 

# Calculate model times from start and end dates
# Add 1 second to end time to ensure time_end is the last value in the time series
mod_datetime = np.arange(time_start, time_end + pd.to_timedelta('1 second'),np.timedelta64(time_step,'m'), dtype='datetime64')
mod_time = np.arange(0,time_step*len(mod_datetime),time_step) # Time starting at 0 in minutes

# Convert UTC model time to PST of test reach
mod_time_conv = pd.Series(mod_datetime)
mod_time_conv = pd.Series(mod_time_conv.dt.to_pydatetime())
mod_time_conv = mod_time_conv.apply(lambda x: x.replace(tzinfo = pytz.utc)) # Add UTC time zone
mod_time_pst = mod_time_conv.apply(lambda x: x.astimezone(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S')) # Convert to EST

mod_time_pst = pd.to_datetime(mod_time_pst)

# Define number of time steps
n_dt = len(mod_time)


## METEOROLOGICAL FORCINGS ## 

# # Extract meteorological and radiative variables (columns = nodes, rows = times)
at = data_reformat(hja_rch_data, 'T2D', time_step) - 273.15 # Air temperature, Units: deg C

# Convert meteorological forcings to model nodes
at = spat_interp(at, reaches_in, nodes_center)

# Extract extended air temperature record
at_extend = data_reformat(ldasin_rch_data, 'T2D', time_step) - 273.15

# Retrieve at_extend time
at_extend_datetime = np.arange(ldasin_rch_data.nwm_time[0], time_end + pd.to_timedelta('1 second'),np.timedelta64(time_step,'m'), dtype='datetime64')

# Convert extended air temperature record to model nodes
# Assign values to model nodes based on their centroid location relative to reach segments
mod_nodes_loc = np.searchsorted(reaches_in[1:], nodes_center)

# Fill in values at model nodes for output based on location of the model nodes
at_extend = at_extend[:,mod_nodes_loc]


## TEMPERATURE INITIATION ##

# Estimate GW temperatures using AT
# Load annual air temperatures from PRISM
hja_prism = pd.read_csv("../NWM-water-temperature/data_formatting/site_data/prism_at/hja_prism_at.csv")

# Filter hja_prism to segment
hja_prism_seg = np.array(hja_prism.at_mean4[hja_prism.segment == segment])

# Spatially interpolate to model nodes
hja_prism_mean = spat_interp(hja_prism_seg, reaches_in, nodes_center)

# # Calculate 7-day rolling average of air temperatures based on air temperature from NWM forcings at each reach
# Create dataframe with air temperatures
at_df = pd.DataFrame(at)
at_extend_df = pd.DataFrame(at_extend)

# Add datetime index
at_df = at_df.set_index(pd.to_datetime(mod_time_pst))

# Add datetime index to at_extend_df, use datetime from first reach for multisegment reaches
at_extend_df = at_extend_df.set_index(pd.to_datetime(at_extend_datetime))

# Resample AT to daily means
at_day = at_df.resample('D').mean()
at_extend_day = at_extend_df.resample('D').mean()

# Calculate 7-day rolling average of air temperatures
#https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/
window = 7 # Convert to hourly step
window_date = []
moving_at = []
i = 0

# Loop throught AT record (extended at record)
while i < len(at_day):
    
    # at_extend_day is 16 indices behind at_day
         
    # Store elements within window
    win = at_extend_day.iloc[(i+17)-window:i+17,:]

    # Calculate average of window and add to list
    moving_at.append(np.mean(win,axis=0))
    
    # Add date of the end of the moving window (incorporates last 3 days of AT)
    window_date.append(at_day.index[i])
    
    # Increment
    i += 1

# Convert moving_at to array
moving_at = np.array(moving_at)
moving_at = np.vstack(moving_at)
window_date = pd.to_datetime(np.array(window_date)).values

# Adjust window_date to align with model time
window_date[0] = pd.to_datetime(mod_time_pst[0])
# Add final timestep to align with model time (model ending time in UTC)
window_date = np.append(window_date,[np.datetime64('2019-07-31 16:00:00')])
# Repeat last row of daily means in moving_at
moving_at = np.vstack([moving_at, moving_at[31,:]])

# Resample moving at to hourly time step
moving_at_df = pd.DataFrame(moving_at)
moving_at_df = moving_at_df.set_index(window_date)
moving_at_hour = moving_at_df.resample('H').interpolate(method='time')

# Calculate difference between moving_at window and estimated mean annual temperature
at_diff = moving_at_hour - hja_prism_mean

# Initialize array to store coefficients at each reach
at_gw_coeff = np.zeros(n)

# Create array of Coefficients
coeff_arr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Create dataframes to store GW temps at along Fa segment
gw_head = pd.DataFrame(np.zeros((len(mod_time_pst),len(coeff_arr))))

# Simulate GW temperature using different AT-GW coefficients
for i in range(0,len(coeff_arr)):

    # Add air temperature forced oscillations to mean annual air temp to estimate GW temperature
    at_gw = coeff_arr[i] * at_diff + hja_prism_mean
    
    # Resample at_gw and at_bc to model time
    at_gw = at_gw.resample(str(time_step)+'T').interpolate(method='time')

    # Extract GW temp at headwater and Lookout Creek gage
    gw_head.iloc[:,i] = at_gw.iloc[:,0].values

# Create plots for varying AT-GW Coefficient: Headwater
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('')
plt.xticks(rotation=90)
plt.ylabel('Inflow Temperature (C)')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.plot(mod_time_pst[7:],at[7:,0],color='red',alpha = 0.25)
plt.ylim((5,27.5))

for i in range(0,len(coeff_arr)):

    # Plot Groundwater temperature lines over time
    plt.plot(mod_time_pst[7:], gw_head.iloc[7:,i],color='black')
    
    # Annotate coefficients for each line
    plt.annotate(str(coeff_arr[i]),(mod_time_pst[743],gw_head.iloc[743,i]),
                 va = 'center', ha = 'left')

# Annotate Air Temperature
plt.annotate(str('AT'),(mod_time_pst[743],at[743,0]+0.5),color='red',alpha=0.25,)

plt.show() 


## GROUNDWATER TEMPERATURES FOR FULL YEAR ##
# This is a rough estimate using observed temperatures from HJ Andrews

# Estimate GW temperatures for AT
# Load observed daily air temperature from HJA
hja_at = pd.read_csv('../NWM-water-temperature/data_formatting/site_data/HT00401.csv')

# Convert DATE to date class
hja_at.DATE = pd.to_datetime(hja_at.DATE, format = "%Y-%m-%d")

# Load annual air temperatures from PRISM
hja_prism = pd.read_csv("../NWM-water-temperature/data_formatting/site_data/prism_at/hja_prism_at.csv")

# Filter hja_prism to segment
hja_prism_seg = np.array(hja_prism.at_mean4[hja_prism.segment == 'Fa'])

# Filter to GSMACK
gsmack_at = hja_at[hja_at.SITECODE == "GSMACK"]

# Filter to date and temperature
mod_time_extend = gsmack_at.iloc[11371:,7]
gsmack_at_extend = gsmack_at.iloc[11371:,8]

# Filter to model time
mod_time = gsmack_at.iloc[11378:,7]

# Filter GSMACK AT to WY 2019
gsmack_at_2019 = gsmack_at.iloc[11378:,8]


# Calculate rolling average of air temperatures
window = 7 # Moving window duration (days)
# window = int(window)
window_date = []
moving_at = []
i = 0

# Loop throught AT record (extended at record)
while i < len(mod_time):
    
    # at_extend_day is 7 indices behind at_day
    
    # Store elements within window
    win = gsmack_at_extend.iloc[(i+7)-window:i+7]

    # Calculate average of window and add to list
    moving_at.append(np.mean(win,axis=0))
    
    # Add date of the end of the moving window (incorporates last 3 days of AT)
    window_date.append(mod_time.iloc[i])
    
    # Increment
    i += 1

# Calculate at_diff from mean air temperature
at_diff = moving_at - hja_prism_seg

# Add air temperature forced oscillations to mean annual air temp to estimate GW temperature
at_gw_0 = 0 * at_diff + hja_prism_seg
at_gw_02 = 0.2 * at_diff + hja_prism_seg
at_gw_04 = 0.4 * at_diff + hja_prism_seg
at_gw_06 = 0.6 * at_diff + hja_prism_seg
at_gw_08 = 0.8 * at_diff + hja_prism_seg
at_gw_10 = 1 * at_diff + hja_prism_seg

# Date formatting
locator = mdates.MonthLocator()
fmt = mdates.DateFormatter('%b-%y')

# Create plot of annual GW temperature signals
plt.figure()
X = plt.gca().xaxis
X.set_major_locator(locator)
X.set_major_formatter(fmt)
plt.plot(mod_time, at_gw_0,c="#00429d",lw=1.5,label="AT-GW = 0")
plt.plot(mod_time, at_gw_02,c="#5681b8",lw=1.5,label="AT-GW = 0.2")
plt.plot(mod_time, at_gw_04,c="#94c3ca",lw=1.5,label="AT-GW = 0.4")
plt.plot(mod_time, at_gw_06,c="#e5b198",lw=1.5,label="AT-GW = 0.6")
plt.plot(mod_time, at_gw_08,c="#c1665b",lw=1.5,label="AT-GW = 0.8")
plt.plot(mod_time, at_gw_10,c="#93003a",lw=1.5,label="AT-GW = 1.0")
plt.ylim([-5,20])
plt.ylabel('Estimated Inflow Temperature (C)')
plt.xticks(rotation = 45)
plt.legend()
plt.show()



