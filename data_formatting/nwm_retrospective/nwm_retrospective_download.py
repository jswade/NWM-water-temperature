# -*- coding: utf-8 -*-
"""
Download and extract NWM v2.1 Retrospective Data from AWS within H.J. Andrews test catchment
for July 2019 study period.

# NWM v2.1 Retrospective available on AWS at:
# https://registry.opendata.aws/nwm-archive/
# (NetCDF format): https://noaa-nwm-retrospective-2-1-pds.s3.amazonaws.com/index.html 

"""

import pandas as pd
import geopandas as gpd
import numpy as np
import pytz
import xarray as xr
import netCDF4
import requests
import boto3
from botocore import UNSIGNED
from botocore.config import Config


## Define model reaches: see /data_formatting/NWM_channels ##

# Load HJA channel dataframe
hja_df = pd.read_csv('../NWM-water-temperature/data_formatting/NWM_channels/formatted_channels/hja_channel.csv')

# Find number of reach segments
n = len(hja_df['feature_id'])

# Convert channel feature_id and lat/lon to geopandas attribute
reach_pt = pd.DataFrame(data={'feature_id':hja_df['feature_id'],'lat':hja_df['lat'], 'lon':hja_df['lon']})

# Convert reach_pt to geodataframe
reach_pt = gpd.GeoDataFrame(reach_pt, crs='EPSG:4269',geometry=gpd.points_from_xy(reach_pt.lon, reach_pt.lat))

# Set proj4 string from forcing file (Lambert Conformal Conical)
lcc_proj4 = '+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 +lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 +k_0=1.0 +nadgrids=@null +wktext  +no_defs '

# Convert reach_pt to LCC CRS
reach_pt_lcc = reach_pt.to_crs(lcc_proj4)

# Add columns to reach_pt_lcc for x_lcc and y_lcc
reach_pt_lcc['x_lcc'] = reach_pt_lcc.geometry.apply(lambda p: p.x)
reach_pt_lcc['y_lcc'] = reach_pt_lcc.geometry.apply(lambda p: p.y)

# Add timezone data
utc = pytz.utc
pst = pytz.timezone('America/Los_Angeles')

# Subset hja_df for segment ids
hja_df_seg = hja_df.iloc[:,29:32]

# Rename column to match hja_df
hja_df_seg = hja_df_seg.rename(columns = {'feat_str':'feature_id'})

# Identify feature_id of HJA reaches
chan_feats = reach_pt['feature_id'][:]

# LDASIN NWM V2.1 Retrospective Files do not have projection info
# Grid cells of LDASIN are defined by sequential west_east and south_north integers
# 4608 x 3840
# These grid cells match the dimensions of those used by forcing NWM operational data, which do have projections in LCC
# Load operational V2.1 NWM test forcing data to copy projection
forcing_test = xr.open_dataset('../NWM-water-temperature/data_formatting/NWM_retrospective/proj_file/nwm.t00z.analysis_assim.forcing.tm00.conus.nc')

# Intersect HJA reach points with netcdf to retrieve values
forcing_test_hja = forcing_test.sel(x=reach_pt_lcc['x_lcc'], y=reach_pt_lcc['y_lcc'],method="nearest")   

# Retrieve x and y values of grid points nearest to HJA river points
hja_x = forcing_test_hja.x.values
hja_y = forcing_test_hja.y.values

# Retrieve all x and y values of WRF-Hydro forcing grid in Lambert Conformal Conical
forcing_lcc_x = forcing_test.x.values
forcing_lcc_y = forcing_test.y.values

# Find the grid cell index from NWM forcing netcdf closest to each test basin reach
x_ind = np.repeat(0,len(hja_x))
y_ind = np.repeat(0,len(hja_y))

for k in range(0,len(hja_x)):
 
    # Find x index of HJA point
    x_ind[k] = np.where(forcing_lcc_x == hja_x[k])[0]
    # Find y index of HJA point
    y_ind[k] = np.where(forcing_lcc_y == hja_y[k])[0]




## AWS Formatting and File Retrieval ##

# Define start and end datetime for NWM file download (NOTE: THESE TIMES ARE IN UTC)
startdate = '2019-07-01T00:00:00'
#startdate = '2019-06014T00:00:00' # Enable to generate extended dataset (for air temperature moving window approach to groundwater inflow temperature)
enddate = '2019-07-31T23:00:00'

# Create datetime array of hourly steps between start and end date in UTC
time_range = pd.date_range(startdate, enddate, freq='H',tz="UTC")

# Find number of hours in record
n_hr = len(time_range)

# Format time_range to match AWS http format
yr_list = time_range.strftime('%Y').values.astype('str')
# CHRTOUT files include hour and minute
chrtout_dt_list = time_range.strftime('%Y%m%d%H%M').values.astype('str')
# LDASIN files include only hour
ldasin_dt_list = time_range.strftime('%Y%m%d%H').values.astype('str')

# Preallocate arrays to store AWS keys
chrtout_keys = []
ldasin_keys = []

# Assemble array of URLs for CHRTOUT and LDASIN files
for i in range(0,len(yr_list)):

    chrtout_keys.append("model_output/" + yr_list[i] + "/" + chrtout_dt_list[i] + ".CHRTOUT_DOMAIN1.comp")

    ldasin_keys.append("forcing/" + yr_list[i] + "/" + ldasin_dt_list[i] + ".LDASIN_DOMAIN1")


# Initialize dataframe to store data at each reach segment
# Number of rows = # of reaches * # of files
hja_chrtout_data = pd.DataFrame(data={'feature_id':np.tile(reach_pt['feature_id'],(n_hr)),
                                'nwm_time':np.zeros(n*n_hr), 'streamflow':np.zeros(n*n_hr),
                                'velocity':np.zeros(n*n_hr),'qSfcLatRunoff':np.zeros(n*n_hr), 
                                'qBucket':np.zeros(n*n_hr)})

hja_ldasin_data = pd.DataFrame(data={'feature_id':np.tile(reach_pt['feature_id'],(n_hr)),
                                'nwm_time':np.zeros(n*n_hr),'U2D':np.zeros(n*n_hr),
                                'V2D':np.zeros(n*n_hr), 'LWDOWN':np.zeros(n*n_hr),
                                'T2D':np.zeros(n*n_hr), 'Q2D':np.zeros(n*n_hr),
                                'PSFC':np.zeros(n*n_hr),'SWDOWN':np.zeros(n*n_hr)})


# ## AWS Access ##
# # See https://github.com/HamedAlemo/visualize-goes16/blob/master/visualize_GOES16_from_AWS.ipynb

# Set bucket name and key
bucket_name = 'noaa-nwm-retrospective-2-1-pds'

# Initialize s3 client.
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Retrieve and store data from CHRTOUT files
for i in range(0,len(chrtout_keys)):
    
    print(i)
    
    ## AWS Retrieval ##
    # Credit: Hamed Alemo, https://github.com/HamedAlemo/visualize-goes16/blob/master/visualize_GOES16_from_AWS.ipynb

    # Index key based on datetime
    key = chrtout_keys[i]
    
    # Get file from AWS Bucket
    resp = requests.get(f'https://{bucket_name}.s3.amazonaws.com/{key}')
    
    # Get file name of retrieved file
    file_name = key.split('/')[-1]
    
    # Initialize NWM dataset into memory
    nc4_ds = netCDF4.Dataset(file_name, memory = resp.content)
    
    # Store netCDF for reading and writing
    store = xr.backends.NetCDF4DataStore(nc4_ds)
    
    # Open netCDF file using xarray
    DS = xr.open_dataset(store)
    
    # Convert dataset to dataframe
    chan_df = DS.to_dataframe()

    ## Subsetting NWM files using feature_ids of test basin ##
    # Slice chan_df by feature_ids
    hja_chan_df = chan_df.loc[:,:,chan_feats]

    # Add channel data to dataframe (indices correspond to j)
    hja_chrtout_data.loc[(11*i):(11*i+10),'nwm_time'] = hja_chan_df.index.get_level_values('time').values
    hja_chrtout_data.loc[(11*i):(11*i+10),'streamflow'] = np.array(hja_chan_df['streamflow'][:])
    hja_chrtout_data.loc[(11*i):(11*i+10),'velocity'] = np.array(hja_chan_df['velocity'][:])
    hja_chrtout_data.loc[(11*i):(11*i+10),'qSfcLatRunoff'] = np.array(hja_chan_df['qSfcLatRunoff'][:])
    hja_chrtout_data.loc[(11*i):(11*i+10),'qBucket'] = np.array(hja_chan_df['qBucket'][:])
    
hja_chrtout_data.to_csv("../NWM-water-temperature/data_formatting/NWM_retrospective/retrospective_files/retro_v21_chrtout.csv", index=False)    
    

# Retrieve and store data from LDASIN files
for i in range(0,len(ldasin_keys)):
    
    print(i)
    
    i = 0
    
    ## AWS Retrieval ##
    # Credit: Hamed Alemo, https://github.com/HamedAlemo/visualize-goes16/blob/master/visualize_GOES16_from_AWS.ipynb

    # Index key based on datetime
    key = ldasin_keys[i]
    
    # Get file from AWS Bucket
    resp = requests.get(f'https://{bucket_name}.s3.amazonaws.com/{key}')
    
    # Get file name of retrieved file
    file_name = key.split('/')[-1]
    
    # Initialize NWM dataset into memory
    nc4_ds = netCDF4.Dataset(file_name, memory = resp.content)
    
    # Store netCDF for reading and writing
    store = xr.backends.NetCDF4DataStore(nc4_ds)
    
    # Open netCDF file using xarray
    forcing_DS = xr.open_dataset(store)
    
    # Index forcing netCDF using x and y indices of HJA reaches
    forcing_hja = forcing_DS.isel(west_east=xr.DataArray(x_ind), south_north=xr.DataArray(y_ind))
    
    # Add channel data to dataframe (indices correspond to j)
    hja_ldasin_data.loc[(11*i):(11*i+10),'nwm_time'] = str(forcing_hja.Times.values)[2:21]
    hja_ldasin_data.loc[(11*i):(11*i+10),'U2D'] = forcing_hja.U2D.values[0]
    hja_ldasin_data.loc[(11*i):(11*i+10),'V2D'] = forcing_hja.V2D.values[0]
    hja_ldasin_data.loc[(11*i):(11*i+10),'LWDOWN'] = forcing_hja.LWDOWN.values[0]
    hja_ldasin_data.loc[(11*i):(11*i+10),'T2D'] = forcing_hja.T2D.values[0]
    hja_ldasin_data.loc[(11*i):(11*i+10),'Q2D'] = forcing_hja.Q2D.values[0]
    hja_ldasin_data.loc[(11*i):(11*i+10),'PSFC'] = forcing_hja.PSFC.values[0]
    hja_ldasin_data.loc[(11*i):(11*i+10),'SWDOWN'] = forcing_hja.SWDOWN.values[0]

hja_ldasin_data.to_csv("../NWM-water-temperature/data_formatting/NWM_retrospective/retrospective_files/retro_v21_ldasin.csv", index=False)    

## Combine data download files into single file

# Read in data if already downloaded
hja_ldasin_data = pd.read_csv("../NWM-water-temperature/data_formatting/NWM_retrospective/retrospective_files/retro_v21_ldasin.csv")    
hja_chrtout_data = pd.read_csv("../NWM-water-temperature/data_formatting/NWM_retrospective/retrospective_files/retro_v21_chrtout.csv")

# Combine dataframes
hja_data = hja_chrtout_data.join(hja_ldasin_data.iloc[:,2:9])

# Inner join segment IDs to feature_ids in data
hja_data = pd.merge(hja_data, hja_df_seg, on='feature_id', how='inner')

# Sort by hja_id
hja_data = hja_data.sort_values(by=['hja_id','nwm_time'])

#### Write dataframe to file ####
hja_data.to_csv('../NWM-water-temperature/data_formatting/NWM_retrospective/retrospective_files/retro_v21_data.csv', index=False)
