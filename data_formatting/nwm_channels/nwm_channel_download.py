# -*- coding: utf-8 -*-
"""

# Download and format NWM channel channel data located within H.J. Andrews Experimental Forest (HJA) test
catchment as input to ST model

Channel data is derived from NWM RouteLink and Hydrofabric datasets

* RouteLink
File description:  This file contains river channel definition and parameter
data, and USGS stream gauge river reach definitions used in all configurations
of the National Water Model. Based on NHDPlus v2.
# Credit: https://www.nohrsc.noaa.gov/pub/staff/keicher/NWM_live/NWM_parameters/README.txt
# Source: https://water.noaa.gov/about/nwm#:~:text=parameter%20files%2C%20click-,here.,-Model%20Parameter%20Files


# NWM Hydrofabric Geodatabase
# Geometry objects defining location and identification of NWM reaches
# https://www.nohrsc.noaa.gov/pub/staff/keicher/NWM_live/web/data_tools/NWM_channel_hydrofabric.tar.gz

"""

# Import Python packages
from shapely import ops
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import fiona
import numpy as np
import xarray as xr


# Read HJ Andrews AOI
# HJ Andrews Watershed defined using StreamStats, a USGS Application
# (https://www.usgs.gov/streamstats)
hja_aoi = gpd.read_file("../NWM-water-temperature/data_formatting/NWM_channels/raw_data/shapefiles/hja_watershed.shp")

# Set CRS
hja_aoi = hja_aoi.set_crs('epsg:4326')

# Calculate aoi bounds
hja_bounds = hja_aoi.total_bounds

# Project hja_aoi to reaches CRS
hja_aoi = hja_aoi.to_crs(epsg=4269)


## Load and format routelink file
# Please download RouteLink_CONUS.nc at https://www.nohrsc.noaa.gov/pub/staff/keicher/NWM_live/NWM_parameters/NWM_parameter_files.tar.gz
# Copy downloaded RouteLink_CONUS.nc file into proper repository folder (see file path below)

# Set filepath of RouteLink
routelink_fp = "../NWM-water-temperature/data_formatting/NWM_channels/raw_data/NWM_parameters_v2.1/RouteLink_CONUS.nc"

# Read RouteLink file
routelink = xr.open_dataset(routelink_fp)
rl_df = routelink.to_dataframe()

# Convert routelink to geopandas geometry
rl_gdf = gpd.GeoDataFrame(rl_df, crs='EPSG:4269',geometry=gpd.points_from_xy(rl_df.lon, rl_df.lat))

# Subset RouteLink points to HJA Study domain
rl_hja = gpd.overlay(rl_gdf,hja_aoi, how="intersection")

# Remove segment 23773409: Connector segment in lake
rl_hja = rl_hja.drop(11, axis=0)

# Add column for string version of Feature ID for subsetting
rl_hja['feat_str'] = rl_hja['link'].astype("string")

# Sort rl_hja by link
rl_hja= rl_hja.sort_values(by=['link'])


## Load and format hydrofabric file
# Set filepath of hydrofabric geodatabase
# Please download NWM_v2.1_channel_hydrofabric at https://www.nohrsc.noaa.gov/pub/staff/keicher/NWM_live/web/data_tools/NWM_channel_hydrofabric.tar.gz
# Copy downloaded NWM_v2.1_channel_hydrofabric file into proper repository folder (see file path below)

hydrofab_fp = "../NWM-water-temperature/data_formatting/NWM_channels/raw_data/NWM_v2.1_channel_hydrofabric/nwm_v2_1_hydrofabric.gdb"

# List layers in hydrofabric file
hydrofab_lay = fiona.listlayers(hydrofab_fp)

# Read CONUS reach file (takes some time to run)
reaches = gpd.read_file(hydrofab_fp, driver='FileGDB', layer=5)

# Subset CONUS reach file to the HJA Study domain
reaches_hja = gpd.overlay(reaches,hja_aoi, how="intersection")

# Add column for string version of Feature ID for subsetting
reaches_hja['feat_str'] = reaches_hja['feature_id'].astype("string")

# Remove segment 23773409: Connector segment in lake
reaches_hja = reaches_hja.drop(11, axis=0)

# Sort reaches_hja by feature_id to align with rl_hja
reaches_hja = reaches_hja.sort_values(by=['feature_id'])


# Combine hydrofabric and routelink files into a single dataframe based on ComID
# Remove rl_hja geometry column
# 'Link' in routelink corresponds to ComID of reach segment
hja_df = pd.concat([reaches_hja.iloc[:,reaches_hja.columns != "feat_str"].reset_index(drop = True), rl_hja.iloc[:,rl_hja.columns != "geometry"].reset_index(drop = True)], axis=1)

# Remove duplicate columns
hja_df = hja_df.loc[:,~hja_df.columns.duplicated()].copy()                   



## Assign unique alphanumeric identifiers to HJA reaches
# Load hja_channel_id.csv
hja_chann_id = pd.read_csv('../NWM-water-temperature/data_formatting/NWM_channels/formatted_channels/hja_channel_id.csv')

# Sort hja_chann_id to line up with hja_df
hja_chann_id = hja_chann_id.sort_values(by='feature_id')

# Set index to match hja_df
hja_chann_id = hja_chann_id.set_index(pd.Index(list(range(0,11))))

# Add hja_id and segment to hja_df to identify order of calculation in model
hja_df['hja_id'] = hja_chann_id['hja_id']
hja_df['segment'] = hja_chann_id['segment']

# Sort ns_df by segment id
hja_df = hja_df.sort_values(by='hja_id',ignore_index=True)

# Write hja_df to csv
hja_df.to_csv('../NWM-water-temperature/data_formatting/NWM_channels/formatted_channels/hja_channel.csv', index=False)



## Determine location of two temperature gages along the stream network
# HJA Gage Locations ('/Users/jswade/user/noaa/hjandrews/hja_data/hja_channels/USGS/hja_gage_locations.csv')
# Credit: Stanley V. Gregory, Sherri L. Johnson, 2019
# https://andlter.forestry.oregonstate.edu/data/abstractdetail.aspx?dbcode=HT004

# Load HJA Gage Locations and Snap Gages to NHD Reaches
# https://gis.stackexchange.com/questions/306838/snap-points-shapefile-to-line-shapefile-using-shapely
# Read in HJA Gage info
hja_gage = pd.read_csv('../NWM-water-temperature/data_formatting/NWM_channels/raw_data/reach_formatting/selected_gage_locations.csv')

# Zip coordinates into pair
hja_gage['Coordinates'] = list(zip(hja_gage.Lon, hja_gage.Lat))

# Transform tuples to point
hja_gage['Coordinates'] = hja_gage['Coordinates'].apply(Point)

# Convert gage info to gdf
hja_gage = gpd.GeoDataFrame(hja_gage, crs="EPSG:4269", geometry=hja_gage['Coordinates'])

# Remove gages not used in ST model
hja_gage = hja_gage.iloc[0:2,:]


# Join NHD lines and Gage points
gage_point = hja_gage.geometry.unary_union
reaches_line = reaches_hja.geometry.unary_union

# Interpolate and project gages point to lines
gage_snap = hja_gage.copy()
gage_snap['geometry'] = gage_snap.apply(lambda row: reaches_line.interpolate(reaches_line.project( row.geometry)), axis=1)

# Update coordinates of gages_snap
gage_snap['Coordinates'] = gage_snap.geometry


## Calculate distance of gages along reaches

# Project lines and gages to WGS 84 UTM Zone 10N (Eugene, Oregon) to calculate distance in meters
reaches_hja = reaches_hja.to_crs('epsg: 32610')
gage_snap = gage_snap.to_crs('epsg: 32610')

# Calculate reach length in meters
reaches_hja['Shape_Length_m'] = reaches_hja.length

# Join Ma and Fa line segments
Ma_line = reaches_hja[reaches_hja.gnis_name == "Lookout Creek"].geometry.unary_union
Fa_line = reaches_hja[reaches_hja.gnis_name == "Mack Creek"].geometry.unary_union

# Merge multiline strings into single lines
Ma_line = ops.linemerge(Ma_line)

# Calculate distance along line
GSLOOK_loc = Ma_line.project(gage_snap.geometry[0]) # GSLOOK (Ma): 15107
GSMACK_loc = Fa_line.project(gage_snap.geometry[1]) # GSMACK (Fa): 3435

# Add locations of gages along reaches to gages_snap
gage_loc = np.array([GSLOOK_loc, GSMACK_loc])
gage_seg = np.array(["Ma", "Fa"])
gage_snap['gage_dist'] = gage_loc
gage_snap['gage_seg'] = gage_seg

# Write snapped points to csv
gage_snap.to_csv('../NWM-water-temperature/data_formatting/NWM_channels/raw_data/reach_formatting/hja_gage_snap.csv')

