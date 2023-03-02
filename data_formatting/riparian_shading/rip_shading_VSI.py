# -*- coding: utf-8 -*-
"""

Estimate shading along H.J. Andrews reaches using gridded US Forest Service LANDFIRE datasets
 
LANDFIRE, (2020). Existing Vegetation Cover Layer, LANDFIRE 2.2.0, U.S. Department of the Interior, Geological Survey, and U.S. Department of Agriculture. accessed 17 October 2022, http://landfire.cr.usgs.gov/viewer.
LANDFIRE, (2020). Existing Vegetation Height Layer, LANDFIRE 2.2.0, U.S. Department of the Interior, Geological Survey, and U.S. Department of Agriculture. accessed 17 October 2022, http://landfire.cr.usgs.gov/viewer.

Implement Kalny et al., 2017 Vegetation Shading Index (VSI)

Kalny, G., Laaha, G., Melcher, A., Trimmel, H., Weihs, P., & Rauch, H. P. (2017). The influence of riparian vegetation shading on water temperature during low flow conditions in a medium sized river. Knowledge and Management of Aquatic Ecosystems, 418. https://doi.org/10.1051/kmae/2016037

"""

# Load libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import math
import rasterio
from rasterio.mask import mask
from shapely import wkt, geometry, ops
from shapely.geometry import LineString
import pytz
import pyproj
import pysolar.solar as solar
# Ignore shapely warnings
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'

# Create function to reformat data from input file into dataframe of variable, where rows=time, columns=nodes
# Temporally resamples data from hourly to model time (in cases where time step choice is not hourly)
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

    # Check dimensions of input (if number of time steps = 744)
    if len(input_val) != 744: # Array is (n_dt,n)
         
        # Fill in values at model nodes for output based on location of the model nodes
        # Single dimension array
        out_val = input_val[mod_nodes_loc]
    
    elif len(input_val) == 744: # Array is (n,)
        
        # Fill in values at model nodes for output based on location of the model nodes
        # Multi-dimension array
        out_val = input_val[:,mod_nodes_loc]
    
    # Convert to array
    out_val = np.array(out_val)
    
    # Return formatted values
    return(out_val)


# Load model forcing and streamflow data (see retrospective_download_hja.py)
hja_data = pd.read_csv('../NWM-water-temperature/data_formatting/NWM_retrospective/retrospective_files/retro_v21_data.csv')

# Load model channel data (see hja_reach_download.py)
hja_df = pd.read_csv('../NWM-water-temperature/data_formatting/NWM_channels/formatted_channels/hja_channel.csv')


## Create function to extract riparian shading data at model nodes using Vegetation Shading Index (Kalny et al., 2017) and LANDFIRE Canopy Cover, Vegetation Height ##
# segment = selected tributary in HJ Andrews network
# node_space = model spatial resolution in meters
# time_step = model temporal resolution in minutes
def shade_calc(segment, node_space, time_step):

    
    # Filter hja_df by reaches 
    hja_rch = hja_df[hja_df['segment'] == segment]
    
    # Sort reaches by node number
    hja_rch = hja_rch.sort_values(by='hja_id')
    
    # Reset indices of hja_rch
    hja_rch = hja_rch.reset_index(drop=True)
    
    # Retrieve list of feature_ids corresponding to selected reach
    hja_rch_data = hja_data[hja_data['feature_id'].isin(np.array(hja_rch['feature_id']))]
    
    
    
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
    
    # If segment is Fa or Ma, add additional model node at observed gage location
    # Prevents error in calculating prediction RMSE due to distance between model node and gage
    if segment == 'Fa':
        
        # Locations of gage: Mack (Fa) = 3435m
        # Add model node at gage locations
        nodes = np.insert(nodes,np.argmax(nodes > 3435),3435)
        
    if segment == 'Ma':
    
        # Locations of gages: Lookout (Ma) = 15107m
        # Add model node at gage locations
        nodes = np.insert(nodes,np.argmax(nodes > 15107),15107)
    
    # Define number of reach segments (one less than the number of reach boundaries)
    # Last model segment will be shorter than node spacing
    n = len(nodes)-1
    
    # Calculate actual node length between node boundaries
    node_length = np.array([t - s for s, t in zip(nodes, nodes[1:])])
    
    # Define center points of reach segments
    # Values of forcing inputs interpolated based on midpoint of reach
    nodes_center = nodes[1:] - 0.5 * node_length
    
    
    ## TEMPORAL DOMAIN ##
    
    # Sort input data to find start and end times
    hja_rch_data = hja_rch_data.assign(nwm_time = pd.to_datetime(hja_rch_data['nwm_time']))
    input_sort = hja_rch_data.sort_values('nwm_time',ignore_index=True)
    nwm_time = input_sort['nwm_time']
    
    # Define model start and end time
    time_start = np.datetime64(nwm_time[0])
    time_end = np.datetime64(nwm_time.iloc[-1]) 
    
    # Calculate model times from start and end dates (5 min)
    # Add 1 second to end time to ensure time_end is the last value in the time series
    mod_datetime = np.arange(time_start, time_end + pd.to_timedelta('1 second'),np.timedelta64(time_step,'m'), dtype='datetime64')

    # Convert UTC model time to PST of test reach
    mod_time_conv = pd.Series(mod_datetime)
    mod_time_conv = pd.Series(mod_time_conv.dt.to_pydatetime())
    mod_time_conv = mod_time_conv.apply(lambda x: x.replace(tzinfo = pytz.utc)) # Add UTC time zone
    mod_time_pst = mod_time_conv.apply(lambda x: x.astimezone(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S')) # Convert to EST
    #mod_time_pst = np.array(mod_time_pst.apply(lambda x: datetime.strptime(x,'%m-%d-%Y %I:%M%p'))) 
    
    mod_time_pst = pd.to_datetime(mod_time_pst)

    
    ## CHANNEL DIMENSIONS ##
    # Solve for channel area and width through time at each node
    # NWM channels are composed of a rectangular top channel and trapezoidal base channel
    
    # Define stream width at each model node
    # Assume no flow in rectangular compound channel
    #top_width_in = hja_rch['TopWdth'] # Units: meters
    bottom_width_in = hja_rch['BtmWdth']
    chslp_in = hja_rch['ChSlp']
    
    # Interpolate channel inputs to model nodes
    ##top_width = spat_interp(top_width_in, reaches_in, nodes_center)
    bottom_width = spat_interp(bottom_width_in, reaches_in, nodes_center)
    chslp = spat_interp(chslp_in, reaches_in, nodes_center)
    

    # Retrieve discharge at each node over time and reformat to model time
    Q_orig = data_reformat(hja_rch_data, 'streamflow', time_step) # Discharge, Units: m3/s
    
    # Interpolate discharge to model nodes
    Q = spat_interp(Q_orig, reaches_in, nodes_center)
    
    # Retrieve velocity at each node and reformat
    V_orig = data_reformat(hja_rch_data, 'velocity', time_step) # Velocity, Units: m/s
    
    # Interpolate velocity to model nodes
    V_in = spat_interp(V_orig, reaches_in, nodes_center)
    
    
    # Calculate area at all nodes and timesteps based on discharge and flow velocity
    chan_area_mat = Q/V_in # Units: meters^2
    
    # Calculate width at all nodes and timesteps based on discharge and flow velocity
    # Based on trapezoidal channel area, assuming no overbank flow
    # Rows = time steps, columns = nodes
    # Based on discharge, not equal to input top width
    chan_width_mat = np.sqrt((4*chan_area_mat/chslp) + (bottom_width ** 2))
    
    # Calculate mean channel width over the period of record
    mean_chan_width = np.mean(chan_width_mat, axis=0)
    
    ## RIPARIAN SHADING ##
    # Define riparian shading and view to sky data at each model node (0-1)
    # Units: Decimal fraction of shading
    # Calculate riparian shading at desired spatial resolution
    
    # Generate LineStrings for line between node boundaries 
    # This ensures canopy cover is centered on model reach, rather than on node boundaries
    
    # Convert to geopandas df
    hja_rch['geometry'] = hja_rch['geometry'].apply(wkt.loads)
    hja_gdf = gpd.GeoDataFrame(hja_rch, crs='EPSG:4269',geometry=hja_rch['geometry'])
    
    # Merge all reach segments
    multi_line_all = geometry.MultiLineString(hja_gdf['geometry'][:].tolist())
    
    # Merge MultiLineString into single reach LineString
    merged_line_all = ops.linemerge(multi_line_all)
    
    # Node distance refers to midpoint of reach segment, first and last nodes are half length (extend past line)
    nodes_reach = nodes.copy()
    
    # Normalize reach distances to length of line in projection
    nodes_reach_proj = nodes_reach/nodes[-1]
    
    # Generate XY coordinates of nodes
    reach_points = [merged_line_all.interpolate(node,normalized=True) for node in nodes_reach_proj]
    
    # Create dataframe of starting and ending reach_points
    reach_pt_df = pd.DataFrame(data={'pt_start':np.zeros(len(reach_points)-1),'pt_end':np.zeros(len(reach_points)-1)})
    
    # Populate dataframe with start and ends of segments
    for pt in range(0,len(reach_points)-1):
        reach_pt_df.iloc[pt,0] = reach_points[pt]
        reach_pt_df.iloc[pt,1] = reach_points[pt+1]
        
    # Create linestrings from points on reach
    reach_lines = reach_pt_df.apply(lambda row: LineString([row['pt_start'], row['pt_end']]), axis=1)
    # Convert to string WKT format
    reach_lines = reach_lines.apply(str).tolist()
    reach_lines = gpd.GeoSeries.from_wkt(reach_lines)
    
    # Convert line strings to pandas dataframe column
    reach_pt_df['geometry'] = reach_lines
    
    # Convert to geopandas dataframe
    reach_pt_gdf = gpd.GeoDataFrame(reach_pt_df, geometry=reach_pt_df['geometry'],crs="EPSG:4269")
    
    # Set proj4 string from forcing file (Albers Conical Equal Area)
    aea_proj4 = 'epsg:5070'
    
    # Convert reach_pt to AEA CRS
    reach_pt_aea = reach_pt_gdf.to_crs(aea_proj4)
    
    # Open LANDFIRE Forest Canopy Cover
    fcc = rasterio.open('../NWM-water-temperature/data_formatting/Riparian_shading/landfire_data/LF2022_CC_220_CONUS/LC22_CC_220.tif')
    
    # Read key of values for FCC
    fcc_key = gpd.read_file('../NWM-water-temperature/data_formatting/Riparian_shading/landfire_data/LF2022_CC_220_CONUS/LC22_CC_220.tif.vat.dbf')
    
    # Open LANDFIRE Existing Vegetation Height
    evh = rasterio.open('../NWM-water-temperature/data_formatting/Riparian_shading/landfire_data/LF2022_EVH_220_CONUS/LC22_EVH_220.tif')
    
    # Read key of values for EVH
    evh_key = gpd.read_file('../NWM-water-temperature/data_formatting/Riparian_shading/landfire_data/LF2022_EVH_220_CONUS/LC22_EVH_220.tif.vat.dbf')
    
    # Retrieve no data values
    nodata_fcc = fcc.nodata
    nodata_evh = evh.nodata
    
    # Set pyproj Geodesic for azimuth calculations
    geodesic = pyproj.Geod(ellps='GRS80') # GRS 1980 from EPSG 5070
    
    # Initialize array to store canopy cover along reach
    cc_reach = np.zeros(n)
     
    # Initialize array to story existing vegetation height along reach
    evh_reach = np.zeros(n)
    
    # Initialize array to store riparian buffer width along raech
    ripwidth_reach = np.zeros(n)   
        
    
    # Loop through reach linestrings to calculate mean canopy cover
    for k in range(0,n):
        
        # Retrieve coordinates from start and end points
        p1 = reach_pt_aea.pt_start[k]
        p2 = reach_pt_aea.pt_end[k]
        
        # Calculate azimuth from start and end points of reaches
        # Azimuth in -180 to 180 range originally
        # https://stackoverflow.com/questions/54873868/python-calculate-bearing-between-two-lat-long
        # J. Taylor, StackOverflow
        fwd_azimuth,back_azimuth,distance = geodesic.inv(p1.x, p1.y, p2.x, p2.y)
            
        # Convert azimuth to 0-360 range
        if fwd_azimuth < 0: # Some westward component
            fwd_azimuth = 360 + fwd_azimuth
            
        # Single sided buffer 
        #https://gis.stackexchange.com/questions/391003/buffering-one-side-of-line-using-geopandas    
            
        # Use azimuth to determine which buffer to create
        if ((fwd_azimuth <= 45) | (fwd_azimuth >= 315)): # N-Flowing River
        
            # Create two-sided buffer around stream centerline (50m + 0.5*river width) with flat end caps
            buffer_poly = reach_pt_aea.geometry[k].buffer(50 + 0.5*mean_chan_width[k], cap_style=2)
            
        if ((fwd_azimuth >= 135) & (fwd_azimuth <= 225)): # S-Flowing River
    
            # Create two-sided buffer around stream centerline (50m + 0.5*river width) with flat end caps
            buffer_poly = reach_pt_aea.geometry[k].buffer(50 + 0.5*mean_chan_width[k], cap_style=2)
            
        if ((fwd_azimuth > 45) & (fwd_azimuth < 135)): # E-Flowing River
            
            # Create one-sided buffer on the right bank around stream centerline (50m + 0.5*river width) with flat end caps
            buffer_poly = reach_pt_aea.geometry[k].buffer(50 + 0.5*mean_chan_width[k],single_sided=True, cap_style=2)
    
        if ((fwd_azimuth > 225) & (fwd_azimuth < 315)): # W-Flowing River
            
            # Create one-sided buffer on the left bank around stream centerline (50m + 0.5*river width) with flat end caps
            buffer_poly = reach_pt_aea.geometry[k].buffer(-1*(50 + 0.5*mean_chan_width[k]),single_sided=True, cap_style=2)
            
        
        # Intersect riparian buffer with Canopy Cover
        # https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal
        out_image_fcc, out_transform_fcc = mask(fcc, [buffer_poly], crop=True)
        
        # Intersect riparian buffer with Vegetation Height
        out_image_evh, out_transform_evh = mask(evh, [buffer_poly], crop=True)
        
        # Retrieve values from masked raster
        fcc_mask = out_image_fcc[0]
        evh_mask = out_image_evh[0]
        
        # Extract values at valid raster locations
        fcc_rip_val = np.extract(fcc_mask != nodata_fcc, fcc_mask)
        evh_rip_val = np.extract(evh_mask != nodata_evh, evh_mask)
        
        # Convert evh to float (for nan assigning)
        evh_rip_val = evh_rip_val.astype(float)
        
        # Assign non-tree height pixels 0 values and open-water 0 values
        evh_rip_val = np.where((evh_rip_val == 11), np.nan, evh_rip_val)
        evh_rip_val = np.where((evh_rip_val < 100) | (evh_rip_val > 138), 0, evh_rip_val)
            
        # Translate evh values to tree height using evh_key
        # EVH values betwen 100-138 correspond to tree heights in meters
        evh_rip_val = evh_rip_val - 100    
        
        # Calculate average value of canopy cover percent for riparian buffer
        cc_reach[k] = np.nanmean(fcc_rip_val)/100
        
        # Calculate average value of existing vegetation height (m) for riparian buffer
        evh_reach[k] = np.nanmean(evh_rip_val)
        
        # Assume full 50m buffer for all reaches
        ripwidth_reach[k] = 50
    
    
    ## Find average elevation of sun during model period from 10am to 2pm
    # https://pysolar.readthedocs.io/en/latest/
    # Set lat/lon of downstream point
    solar_lat = reach_pt_aea.pt_end[n-1].y
    solar_lon = reach_pt_aea.pt_end[n-1].x
    
    # Convert UTC time to datetime
    mod_time_utc = mod_time_conv.dt.to_pydatetime()
    
    # Retrieve all time indices between 10am and 2pm Pacific Time
    time_vals = np.array([10, 11, 12, 13, 14])
    time_ind = np.where(np.in1d(mod_time_pst.dt.hour,time_vals))[0]
    
    # Retrieve UTC time of desired time period
    mod_time_utc_shade = mod_time_utc[time_ind]
    
    # Loop through 10am-2pm time period, calculating solar elevation angle
    solar_elev = np.zeros(len(mod_time_utc_shade))
    for w in range(0,len(mod_time_utc_shade)):
            solar_elev[w] = solar.get_altitude(solar_lat, solar_lon, mod_time_utc_shade[w])
    
    # Calculate mean solar elevation angle in degrees above horizon
    mean_solar_elev = np.mean(solar_elev)
    
    # Calculate riparian shading-stream width coefficient for Kalny et al., 2017 VSI
    rip_coeff = math.tan(mean_solar_elev * math.pi / 180)
    
    ## Calculate Vegetation-Shading Index (VSI) - Kalny et al., 2017
    # Calculate relative vegetation height
    hr = (evh_reach*100)/(mean_chan_width*rip_coeff)
    
    # If relative vegetation height is greater than 100 (shades entire stream), set value to 100
    hr[hr > 100] = 100
    
    # Calculate VSI (Kalny et al., 2017)
    vsi = (1/3) * ((hr/100) + (ripwidth_reach/50) + (cc_reach/1))
    
    # Convert vsi to dataframe
    vsi_df = pd.DataFrame(vsi)

    return(vsi_df)

# Run shading function on each reach segment
vsi_Ma = shade_calc('Ma', 1000, 60)
vsi_Sa = shade_calc('Sa', 1000, 60)
vsi_Fa = shade_calc('Fa', 1000, 60)
vsi_Fb = shade_calc('Fb', 1000, 60)
vsi_Fc = shade_calc('Fc', 1000, 60)
vsi_Fd = shade_calc('Fd', 1000, 60)

# Write shading to file
vsi_Ma.to_csv('/NWM-water-temperature/data_formatting/Riparian_shading/shading_output/vsi_Ma.csv',index=False)
vsi_Sa.to_csv('/NWM-water-temperature/data_formatting/Riparian_shading/shading_output/vsi_Sa.csv',index=False)
vsi_Fa.to_csv('/NWM-water-temperature/data_formatting/Riparian_shading/shading_output/vsi_Fa.csv',index=False)
vsi_Fb.to_csv('/NWM-water-temperature/data_formatting/Riparian_shading/shading_output/vsi_Fb.csv',index=False)
vsi_Fc.to_csv('/NWM-water-temperature/data_formatting/Riparian_shading/shading_output/vsi_Fc.csv',index=False)
vsi_Fd.to_csv('/NWM-water-temperature/data_formatting/Riparian_shading/shading_output/vsi_Fd.csv',index=False)

