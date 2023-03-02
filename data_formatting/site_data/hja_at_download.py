# -*- coding: utf-8 -*-
"""
Download PRISM Data to calculate annual average air temperature at each HJ Andrews Reach over last 4 years

Used to estimate GW temperatures

Credit: PRISM
PRISM Climate Group, Oregon State University, https://prism.oregonstate.edu, data created Dec 2022, accessed 4 Oct 2022.

"""

# Import libraries
import pandas as pd
import geopandas as gpd
from geopandas import points_from_xy
import numpy as np
import rasterio
from shapely import wkt

### PRISM Opening without GDAL 
#https://pymorton.wordpress.com/2016/02/26/plotting-prism-bil-arrays-without-using-gdal/
### Credit: PyMorton

# Read HJA channel data
hja_chan = pd.read_csv("../NWM-water-temperature/data_formatting/nwm_channels/formatted_channels/hja_channel.csv")

# Convert to geopandas df
hja_chan['geometry'] = hja_chan['geometry'].apply(wkt.loads)
hja_gdf = gpd.GeoDataFrame(hja_chan, crs='EPSG:4269',geometry=hja_chan.geometry)
hja_gdf_pt = gpd.GeoDataFrame(hja_chan, crs='EPSG:4269',geometry=points_from_xy(hja_chan.lon, hja_chan.lat))

# Extract point coordinates
hja_coord = [(x,y) for x, y in zip(hja_gdf_pt.lon, hja_gdf_pt.lat)]

# Set location of PRISM file pathes
prism_path_2019 = '../NWM-water-temperature/data_formatting/site_data/prism_at/PRISM_tmean_stable_4kmM3_2019_bil/PRISM_tmean_stable_4kmM3_2019_bil.bil'
prism_path_2018 = '../NWM-water-temperature/data_formatting/site_data/prism_at/PRISM_tmean_stable_4kmM3_2018_bil/PRISM_tmean_stable_4kmM3_2018_bil.bil'
prism_path_2017 = '../NWM-water-temperature/data_formatting/site_data/prism_at/PRISM_tmean_stable_4kmM3_2017_bil/PRISM_tmean_stable_4kmM3_2017_bil.bil'
prism_path_2016 = '../NWM-water-temperature/data_formatting/site_data/prism_at/PRISM_tmean_stable_4kmM3_2016_bil/PRISM_tmean_stable_4kmM3_2016_bil.bil'

# Read PRISM projection
prj = open("../NWM-water-temperature/data_formatting/site_data/prism_at/PRISM_tmean_stable_4kmM3_2019_bil/PRISM_tmean_stable_4kmM3_2019_bil.prj").read()

# Open PRISM data as raster
prism_rast_2019 = rasterio.open(prism_path_2019, crs='EPSG:6269')
prism_rast_2018 = rasterio.open(prism_path_2018, crs='EPSG:6269')
prism_rast_2017 = rasterio.open(prism_path_2017, crs='EPSG:6269')
prism_rast_2016 = rasterio.open(prism_path_2016, crs='EPSG:6269')

# Extract values from prism_raster at hja reaches
hja_gdf['at_2019'] = [x[0] for x in prism_rast_2019.sample(hja_coord)]
hja_gdf['at_2018'] = [x[0] for x in prism_rast_2018.sample(hja_coord)]
hja_gdf['at_2017'] = [x[0] for x in prism_rast_2017.sample(hja_coord)]
hja_gdf['at_2016'] = [x[0] for x in prism_rast_2016.sample(hja_coord)]

# Plot annual values along network
hja_gdf.plot(column = 'at_2019')
hja_gdf.plot(column = 'at_2018')
hja_gdf.plot(column = 'at_2017')
hja_gdf.plot(column = 'at_2016')

# Calculate 4-year annual mean
hja_gdf['at_mean4'] = np.mean(hja_gdf.iloc[:,32:36], axis=1)

# Write to csv
hja_gdf.to_csv("../NWM-water-temperature/data_formatting/site_data/prism_at/hja_prism_at.csv", index=False)
