
# -*- coding: utf-8 -*-
"""

Model function called by nwm_st_model_cal_runs.py and nwm_st_model_val_runs.py

Simulates water temperatures in the H.J. Andrews catchment during specified time period

Model simulates one selected river reach at a time. To model the full network, the function must be executed multiple times from headwater reaches to outlet

"""

# Create function to model temperatures along selected river reach
# Takes in values for segment ID, riparian shading  and T_t0 of the same length as segment ID

def nwm_st(segment, node_space, time_step,mc_sample,mode, Fa=None, Fb=None, Fc=None, Fd=None, Sa=None):
    
    
    # Inputs:
    # segment = selected reach for model run (Ma, Sa, Fa, Fb, Fc, Fc: arbitrary identifiers, see /NWM-water-temperature/data_formatting/NWM_channels/formatted_channels/hja_channels_diagram.png)
    # node_space = desired model reach spacing, meters
    # time_step = desired temporal resolution, minutes
    # mc_sample = input of Monte Carlo sampled parameters for calibration (see nwm_st_model_runs.py)
    # Optional inputs: Fa, Fb, Fc, Fd, Sa etc. = temperature and discharge at last node of each connecting tributary

    # Load libraries
    import pandas as pd
    import numpy as np
    from scipy.interpolate import lagrange
    import pytz
    # Ignore shapely warnings
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
    pd.options.mode.chained_assignment = None  # default='warn'

    # Create function to reformat data from input file into dataframe of variable, where rows=time, columns=nodes
    # Temporally resamples data to model time
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
    
    # Create function to assign inputs from NWM reaches to model reaches
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
    
    
    ## Create function for calculating radiative heat fluxes at a timestep over all nodes ##
    def rad_flux(sw_in, lw_in, at_in, st_in, spec_h_in, v_wind, shade, pa):
        
        
        # sw_in = sw_in[:,5]
        # lw_in = lw_in[:,5]
        # at_in = at[:,5]
        # st_in = T_sim[:,5]
        # spec_h_in = spec_h[:,5]
        # v_wind = wind[:,5]
        # shade = rip_shade
        # pa = air_press[:,5]
        
    
        ## Shortwave Radiation ##
        # Does not account for diffusive solar radiation (see Westhoff, 2007)
        # Function accepts column (one time step) of incoming SW and longitudinal shading values
    
        # Define albedo (Glose, 2013; Magnusson et al., 2012)
        albedo = 0.04
        
        # Adjusts incoming SW flux using albedo and riparian shading
        # (Magnusson et al., 2012; Maidment, 1993; Glose et al., 2017)
        sw_flux = (1-albedo)*(1 - shade) * sw_in # Units: W/m2
    
        ### Longwave Radiation ###
        # Calculates net longwave radiation using Stefan-Boltzman  Law
        # LWnet = LW(atmos) - LW(emitted) + LW(landcover)
        # Function accepts column (one time step) of air temp, cloud cover,
        # riparian shading, relative humidity, and water temperature
        #### Credit: Glose et al., 2017 
    
        ## Atmospheric LW ##
        # Calculate saturated vapor pressure (Dingman, 1994)
        es = 0.611 * np.exp((17.27*at_in)/(237.2+at_in)) # Units: kPa
        
        # Calculate actual vapor pressure from specific humidity (Dingman, 1994; Maidment, 1993)
        ea = ((spec_h_in * pa)/(0.622 + 0.378 * spec_h_in)) # Units: kPa
        
        #  Make sure that RH is a percentage, not a decimal
        rh_in = 100*(ea/es) # Units: percent
        
        # For RH > 97%, Bowen's ratio approaches infinity, resulting in implausible sensible heat
        # # If RH > 97%, reduce RH to equal 97% to prevent anomalous sensible heat
        if rh_in > 97: 
            rh_in = 97

        ## Atmospheric LW ##
        lw_atmos = 0.96 * lw_in  * (1-shade)
    
        ## Emitted LW (Back) ##
        # Note that this is negative (stream loses heat)
        # (Westhoff et al., 2007; Boyd and Kasper, 2003)
        lw_back = -0.96 * 5.67E-8 * ((st_in + 273.2)**4) # Units: W/m2
    
    
        # ## Landcover LW ##
        # Radiation emitted by surrounding landcover based on air temperature
        # (Westhoff et al., 2007; Boyd and Kasper, 2003)
        # (McCutcheon, 1989)
        # Two 0.96 values because we need to account for emissivity of water and of the surrounding LC
        lw_landcov = 0.96 * shade * 0.96 * 5.67E-8 * ((at_in + 273.2)**4) # Units: W/m2: Include landcover
    
        ## Net Longwave Radiation ##
        # Note that lw_back is negative
        lw_flux = lw_atmos + lw_back + lw_landcov # Units: W/m2: 
        
            
        ## Latent Heat - Evaporation ##
        # Calculates latent heat flux using Penman Evaporation
        # (Dingman, 2002; Westhoff et al., 2007)
    
        # Set constant values: Glose, 2013; Westhoff et al., 2007
        # Density of air
        rho_air = 1.2041 # Units: kg/m3
        rho_water = 1000 # Units: kg/m3
        c_air = 1004 # Units: J/kg*degC
        
        # Calculate aerodynamic resistance
        # Westhoff et al., 2007; 
        # These coefficients can be adjusted for each site (GLose et al., 2017)
        ra = 245 / ((0.54 * v_wind) + 0.5) # Units: s/m
        
        # Calculate slope of saturated vapor pressure curve (Maidment, 1993; Glose, 2017)
        # Units: kPa/degC
        vp_slope = (4100 * es)/((237 + at_in)**2)
        
        # Calculate latent heat of vaporization (Westhoff et al., 2007)
        Le = 1000000 * (2.501 - (0.002361 * st_in)) # J/kg
        
        # Calculate psychometric constant (Cuenca, 1989; Westhoff, 2007)
        psych = (c_air * pa)/(0.62198 * Le) # Units: J/kg*degC
        
        # Calculate Penman Evaporation
        # (Maidment,1993; Westhoff et al., 2007, Glose et al, 2017)
        # Units: m/s
        E = ((vp_slope * (lw_flux + sw_flux))/(rho_water * Le * (vp_slope + psych))) +\
               ((c_air * rho_air * psych * (es - ea))/(rho_water * Le * ra * (vp_slope + psych)))
               
        # Calculate latent heat flux (Westhoff, 2007)
        # Note that latent heat is negative when evaporation rate is positive (stream loses heat)
        lh_flux = -1 * rho_water * Le * E # Units: W/m2
    
        ## Sensible Heat ##
        # Calculate convection flux using Bowen Ratio (ratio of convection to evaporation flux)
        # (Bowen. 1926; Westhoff, 2007; Bedient and Huber, 1992)
    
        # Calculate saturated vapor pressure of evaporating surface based on water temperature (Glose et al., 2017; Dingman, 1994)
        es_water = 0.611 * np.exp((17.27*st_in)/(237.3+st_in)) # Units: kPa
        
        # Calculate actual vapor pressure of evaporating surface (Dingman, 1994; Maidment, 1993)
        #  Make sure that RH is a percentage, not a decimal
        ea_water = (rh_in/100) * es_water # Units: kPa
    
        # Calculate Bowen's Ratio (Westhoff, 2007; Glose et al., 2017)
        br = 0.00061 * pa * ((st_in - at_in)/(es_water - ea_water))
    
        # Calculate sensible heat flux (Westhoff, 2007 and others)
        # Note: Positive sensible heat is a heat gain to the stream
        sh_flux = br * lh_flux # Units: W/m2
        
        # Calculate total radiative flux
        net_flux = sw_flux + lw_flux + lh_flux + sh_flux
    
        # Return tuple of total flux, and flux components
        return (net_flux, sw_flux, lw_flux, lw_atmos, lw_back, lw_landcov, lh_flux, sh_flux, E)
    
    
    ## Model Definition ##
    
    # Depending on model configuration, mc_sample (Monte Carlo parameter sample) will have different numbers of variables
    if len(mc_sample) == 5: # M1
    
        # Monte Carlo Sample
        at_gw1 = mc_sample[0]
        at_gw2 = mc_sample[1]
        at_gw3 = mc_sample[2]
        window = mc_sample[3]
        rip = mc_sample[4]
        
        # Set unused values to defaults
        gw1 = 1.0
        gw2 = 1.0
        gw3 = 1.0
        hyp_lag = 2 # Set to 2 days (1 day breaks code)
        hyp_frac1 = 0 # Hyporheic flow to 0
        hyp_frac2 = 0 # Hyporheic flow to 0
        hyp_frac3 = 0 # Hyporheic flow to 0
  
    elif len(mc_sample) == 8: # M2

        # Monte Carlo Sample
        at_gw1 = mc_sample[0]
        at_gw2 = mc_sample[1]
        at_gw3 = mc_sample[2]
        window = mc_sample[3]
        rip = mc_sample[4]
        gw1 = mc_sample[5]
        gw2 = mc_sample[6]    
        gw3 = mc_sample[7]
        
        # Set unused values to defaults
        hyp_lag = 2 # Set to 2 days (1 day breaks code)
        hyp_frac1 = 0 # Hyporheic flow to 0
        hyp_frac2 = 0 # Hyporheic flow to 0
        hyp_frac3 = 0 # Hyporheic flow to 0

    elif len(mc_sample) == 9: # M3
    
        # Monte Carlo Sample
        at_gw1 = mc_sample[0]
        at_gw2 = mc_sample[1]
        at_gw3 = mc_sample[2]
        window = mc_sample[3]
        rip = mc_sample[4]
        hyp_lag = mc_sample[5]
        hyp_frac1 = mc_sample[6]
        hyp_frac2 = mc_sample[7]
        hyp_frac3 = mc_sample[8]
        
        # Set unused values to defaults
        gw1 = 1.0
        gw2 = 1.0
        gw3 = 1.0
        
    elif len(mc_sample) == 12: # M4
    
        # Monte Carlo Sample
        at_gw1 = mc_sample[0]
        at_gw2 = mc_sample[1]
        at_gw3 = mc_sample[2]
        window = mc_sample[3]
        rip = mc_sample[4]
        gw1 = mc_sample[5]
        gw2 = mc_sample[6]    
        gw3 = mc_sample[7]
        hyp_lag = mc_sample[8]
        hyp_frac1 = mc_sample[9]
        hyp_frac2 = mc_sample[10]
        hyp_frac3 = mc_sample[11]  
    
    # Load model forcing and streamflow data (see nwm_retrospective_download.py)
    hja_data = pd.read_csv('../NWM-water-temperature/data_formatting/nwm_retrospective/retrospective_files/retro_v21_data_'+mode+'.csv')

    # Load extended LDASIN forcing data for June (used to calculate GW temperature during beginning on July)
    ldasin_extend = pd.read_csv('../NWM-water-temperature/data_formatting/nwm_retrospective/retrospective_files/retro_v21_ldasin_extend_'+mode+'.csv')

    # Load model channel data (see nwm_channel_download.py)
    hja_df = pd.read_csv('../NWM-water-temperature/data_formatting/nwm_channels/formatted_channels/hja_channel.csv')

    # Filter hja_df by selected model reach
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

    # Initialize array of node distances
    reaches_in = np.zeros(len(hja_rch)+1) 
    
    # Define location of Eulerian model grid
    for p in range(0,len(hja_rch)-1):
        reaches_in[p+1] = hja_rch['Length'][p] + reaches_in[p]
    reaches_in[len(hja_rch)] = hja_rch['Length'][len(hja_rch)-1] + reaches_in[len(hja_rch)-1]

    # Define number of input reaches
    n_in = len(hja_rch)

    # Specify distances of segment boundaries (locations where ST is to be predicted)
    nodes = np.arange(0,reaches_in[n_in], node_space)

    # Add final node to ensure model nodes are the same length as the input reaches
    nodes = np.append(nodes,reaches_in[-1])
    
    # If segment is Fa or Ma, add additional model node at observed gage location
    # Prevents error in calculating prediction RMSE due to distance between model node and gage
    if segment == 'Fa':
        
        # Locations of gage: Mack (Fa) = 3435m from headwater initiation
        # Add model node at gage locations
        nodes = np.insert(nodes,np.argmax(nodes > 3435),3435)
        
    if segment == 'Ma':

        # Locations of gages: Lookout (Ma) = 15107m from headwater initiation
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

    # Calculate model times from start and end dates (5 min)
    # Add 1 second to end time to ensure time_end is the last value in the time series
    mod_datetime = np.arange(time_start, time_end + pd.to_timedelta('1 second'),np.timedelta64(time_step,'m'), dtype='datetime64')
    mod_time = np.arange(0,time_step*len(mod_datetime),time_step) # Time starting at 0 in minutes

    # Convert UTC model time to PST of test reach
    mod_time_conv = pd.Series(mod_datetime)
    mod_time_conv = pd.Series(mod_time_conv.dt.to_pydatetime())
    mod_time_conv = mod_time_conv.apply(lambda x: x.replace(tzinfo = pytz.utc)) # Add UTC time zone
    mod_time_pst = mod_time_conv.apply(lambda x: x.astimezone(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S')) # Convert to EST
    #mod_time_pst = np.array(mod_time_pst.apply(lambda x: datetime.strptime(x,'%m-%d-%Y %I:%M%p'))) 
    mod_time_pst = pd.to_datetime(mod_time_pst)

    # Define number of time steps
    n_dt = len(mod_time)


    ## CHANNEL DIMENSIONS ##
    # Solve for channel area and width through time at each node
    # NWM channels are composed of a rectangular top channel and trapezoidal base channel

    # Define stream width at each model node
    # Assume no flow in rectangular compound channel
    bottom_width_in = hja_rch['BtmWdth']
    chslp_in = hja_rch['ChSlp']

    # Interpolate channel inputs to model nodes
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


    ## CHANNEL AREA CALCULATION DOES NOT INCLUDE OVERBANK FLOW ##

    # Calculate area at all nodes and timesteps based on discharge and flow velocity
    chan_area_mat = Q/V_in # Units: meters^2

    # Calculate channel area of original input reaches
    chan_area_mat_orig = Q_orig/V_orig

    # Calculate width at all nodes and timesteps based on discharge and flow velocity
    # Based on trapezoidal channel area, assuming no overbank flow
    # Rows = time steps, columns = nodes
    # Based on discharge, not equal to input top width
    chan_width_mat = np.sqrt((4*chan_area_mat/chslp) + (bottom_width ** 2))

    # Calculate depth of water based on top and bottom widths
    chan_depth_mat = 0.5 * chslp * (chan_width_mat - bottom_width)

    ## Compute volume of water in each reach segment (distance between nodes * area) over time ##
    # Volume nodes defined between ends of reach segments, such that node is centered on reach midpoint
    # Calculate volume of reach segments
    vol_node = node_length * chan_area_mat # Units: m^3

    # Calculate volume of original input reach segments # Units: m^3
    vol_reach_orig = chan_area_mat_orig * np.array(hja_rch['Length'])

    # Read air pressure and convert from Pa to kPa
    air_press = 0.001 * data_reformat(hja_rch_data, 'PSFC', time_step)

    # Interpolate air pressure to model nodes
    air_press = spat_interp(air_press, reaches_in, nodes_center)

    ## RIPARIAN SHADING ##
    
    # Read Kalny et al., 2017 Vegetation Shading Index values by reach
    rip_shade = pd.read_csv('../NWM-water-temperature/data_formatting/riparian_shading/shading_output/vsi_'+segment+'.csv')
    rip_shade = np.array(rip_shade)

    ## Surface/Subsurface DISCHARGE ##
    # Surface Runoff and GW Discharge can not be interpolated directly to model nodes (conservation of volume)
    # First, calculate the ratio of surface/gw flow to the volume of the original reaches
    # Then, interpolate ratio of surface/gw flow to new nodes
    # Then, calculate fluxes of surface/gw flow to new nodes using new node volumes

    # Read surface flow and groundwater discharge at each reach 
    Q_surf_orig = data_reformat(hja_rch_data, 'qSfcLatRunoff', time_step) # Surface Runoff: Units: m3/s
    Q_gw_orig = data_reformat(hja_rch_data, 'qBucket', time_step) # Groundwater: Units: m3/s

    # Calculate ratio of gw inflow and surface water inflow to original reach volumes
    GW_frac_orig = Q_gw_orig/vol_reach_orig
    Surf_frac_orig = Q_surf_orig/vol_reach_orig

    # Spatially interpolate discharge fractions to all nodes
    GW_frac_node = spat_interp(GW_frac_orig, reaches_in, nodes_center)
    Surf_frac_node = spat_interp(Surf_frac_orig, reaches_in, nodes_center)

    # Calculate GW inflow and surface runoff from fractions and new node volumes: Units: m3/s
    Q_gw = GW_frac_node * vol_node
    Q_surf = Surf_frac_node * vol_node


    ## Model Calibration using Monte Carlo Samples ##
    
    # Tune riparian shading
    rip_shade = rip_shade * rip
    
    # If ripshading is > 1 set to 1
    rip_shade[rip_shade > 1] = 1
 
    # Tune GW inflow independently by stream order
    Q_gw[:,stream_order == 1] = Q_gw[:,stream_order == 1] * gw1
    Q_gw[:,stream_order == 2] = Q_gw[:,stream_order == 2] * gw2
    Q_gw[:,stream_order == 3] = Q_gw[:,stream_order == 3] * gw3

    # Read temperature and discharge of tributary inputs depending on reach segment
    # Sa Connections: Fc (4487 m: Sa)
    # Ma Connections: Sa (9905 m: Ma), Fa (5586 m: Ma), Fb (4680 m: Ma), Fd (11665 m: Ma)
    if segment == 'Fa' or segment == 'Fb' or segment == 'Fc' or segment == 'Fd':
        
        # Headwaters, no tributary inflows
        
        # Create blank matrix of tributary inflows
        Q_trib = np.zeros((n_dt,n))   
        
        # Create matrix of tributary inflow temperatures
        T_trib = np.zeros((n_dt,n))
        
    if segment == 'Sa':
        
        # Find model node where tributary connects to reach
        trib_Fc_node = np.argmin(np.abs(nodes - 4487))

        # Create matrix of tributary inflows
        Q_trib = np.zeros((n_dt,n))   
        
        # Insert discharge of tributary at correct node
        Q_trib[:,trib_Fc_node] = Fc[:,1]
        
        # Create matrix of tributary inflow temperatures
        T_trib = np.zeros((n_dt,n))
        
        # Insert temperature of tributary at correct node
        T_trib[:,trib_Fc_node] = Fc[:,0]
        
    if segment == 'Ma':
        
        # Find model node where tributaries connect to reach
        trib_Ma_node1 = np.argmin(np.abs(nodes - 9905)) # Sa
        trib_Ma_node2 = np.argmin(np.abs(nodes - 5586)) # Fa
        trib_Ma_node3 = np.argmin(np.abs(nodes - 4680)) # Fb
        trib_Ma_node4 = np.argmin(np.abs(nodes - 11665)) # Fd
        
        # Create matrix of tributary inflows
        Q_trib = np.zeros((n_dt,n)) 
        
        # Insert discharge of tributary at correct nodes
        Q_trib[:,trib_Ma_node1] = Sa[:,1]
        Q_trib[:,trib_Ma_node2] = Fa[:,1]
        Q_trib[:,trib_Ma_node3] = Fb[:,1]
        Q_trib[:,trib_Ma_node4] = Fd[:,1]
        
        # Create matrix of tributary inflow temperatures
        T_trib = np.zeros((n_dt,n))
        
        # Insert temperature of tributary at correct node
        T_trib[:,trib_Ma_node1] = Sa[:,0]
        T_trib[:,trib_Ma_node2] = Fa[:,0]
        T_trib[:,trib_Ma_node3] = Fb[:,0]
        T_trib[:,trib_Ma_node4] = Fd[:,0]
        

    # Sum total lateral inflows
    Q_L = Q_surf + Q_trib + Q_gw # Total lateral discharge, Units: m/3s

    ## METEOROLOGICAL FORCINGS ## 

    # Extract meteorological and radiative variables (columns = nodes, rows = times)
    sw_in = data_reformat(hja_rch_data, 'SWDOWN', time_step) # Incoming Shortwave Rad, Units: W/m2
    lw_in = data_reformat(hja_rch_data, 'LWDOWN', time_step) # Surface downward longwave Rad, Units: W/m2
    at = data_reformat(hja_rch_data, 'T2D', time_step) - 273.15 # Air temperature, Units: deg C
    spec_h = data_reformat(hja_rch_data, 'Q2D', time_step) # Specific Humidity, Units: %
    wind_vdir = data_reformat(hja_rch_data, 'V2D', time_step) # V-component Wind speed, Units: m/s
    wind_udir = data_reformat(hja_rch_data, 'U2D', time_step) # U-component Wind speed, Units: m/s
    wind = np.sqrt(np.square(wind_vdir) + np.square(wind_udir)) # Total wind speed, Units: m/s

    # Convert meteorological forcings to model nodes
    sw_in = spat_interp(sw_in, reaches_in, nodes_center)
    lw_in = spat_interp(lw_in, reaches_in, nodes_center)
    at = spat_interp(at, reaches_in, nodes_center)
    spec_h = spat_interp(spec_h, reaches_in, nodes_center)
    wind = spat_interp(wind, reaches_in, nodes_center)

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

    ## Estimate temporally GW temperatures using lagged and buffered air temperatures
    # Load annual air temperatures at each reach derived from PRISM
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
    
    # Calculate X-day rolling average of air temperatures
    window = int(window)
    window_date = []
    moving_at = []
    i = 0

    # Loop throught AT record (extended AT record)
    while i < len(at_day):
        
        # Note: at_extend_day is 16 indices behind at_day
             
        # Store elements within window
        win = at_extend_day.iloc[(i+17)-window:i+17,:]
    
        # Calculate average of window and add to list
        moving_at.append(np.mean(win,axis=0))
        
        # Add date of the end of the moving window (incorporates last X days of AT)
        window_date.append(at_day.index[i])
        
        # Increment
        i += 1

    # Convert moving_at to array
    moving_at = np.array(moving_at)
    moving_at = np.vstack(moving_at)
    window_date = pd.to_datetime(np.array(window_date)).values

    # Adjust window_date to align with model time
    window_date[0] = pd.to_datetime(mod_time_pst[0])
    # Add final timestep to align with model time (model end time in PST)
    window_date = np.append(window_date,[window_date[-1] + np.timedelta64(16,'h')])
    # Repeat last row of daily means in moving_at
    moving_at = np.vstack([moving_at, moving_at[len(moving_at)-1,:]])
    
    # Resample moving at to hourly time step
    moving_at_df = pd.DataFrame(moving_at)
    moving_at_df = moving_at_df.set_index(window_date)
    moving_at_hour = moving_at_df.resample('H').interpolate(method='time')

    # Calculate difference between moving window temperatures and annual mean temperature
    at_diff = moving_at_hour - hja_prism_mean
    
    # Initialize array to store coefficients at each reach
    at_gw_coeff = np.zeros(n)

    # Set air temperature coefficients for 1st, 2nd, and 3rd order streams
    at_gw_coeff[stream_order == 1] = at_gw1
    at_gw_coeff[stream_order == 2] = at_gw2
    at_gw_coeff[stream_order == 3] = at_gw3

    # Add air temperature forced oscillations to mean annual air temp to estimate GW temperature
    at_gw = at_gw_coeff * at_diff + hja_prism_mean

    # Add air temperature forced oscillations to mean annual air temp to estimate Boundary Condition temperature
    at_bc = at_gw_coeff * at_diff + hja_prism_mean
    
    # Repeat last value in at_bc to shift boundary conditions from reach center to T nodes
    at_bc['rep'] = at_bc.iloc[:,n-1]

    # Resample at_gw and at_bc to model time
    at_gw = at_gw.resample(str(time_step)+'T').interpolate(method='time')
    at_bc = at_bc.resample(str(time_step)+'T').interpolate(method='time')

    # Set initial starting temperatures at all nodes at the first time step
    T_t0 = at_bc.iloc[0,:]

    # Set initial upstream boundary condition at upstream node over full model period
    T_n0 = at_bc.iloc[:,0]

    # Create matrix to store simulated temperatures at all nodes and times
    T_sim = np.zeros((n_dt,len(nodes)), dtype=float)

    # Assign boundary conditions to matrix
    T_sim[0,:] = T_t0
    T_sim[:,0] = T_n0


    ## TEMPERATURE OF LATERAL INFLOWS ##

    # Set temperatures of lateral inputs
    # Adapt to account for longitudinal changes in meteorological forcings
    # Credit: Wanders et al., 2019
    T_q_surf = at - 1.5 # Surface runoff 1.5 deg C less than AT, to account for cooling of rain
    
    # Convert groundwater temperatures to array
    T_q_gw = np.array(at_gw)

    # 
    # Catch errors when Q_L = 0
    if np.all(Q_L == 0):
        T_L = np.zeros((n_dt,n))
    else:
        
        # Calculate net temperature of lateral inflows using proportions of total lateral flow
        # Credit: Glose et al., 2013
        if segment == 'Fa' or segment == 'Fb' or segment == 'Fc' or segment == 'Fd':
        
            # No tributary inflows on headwaters
            T_L = ((Q_surf/Q_L) * T_q_surf) + ((Q_gw/Q_L) * T_q_gw)     
        
        else: # segment == 'Sa', segment == 'Ma'
            
            # Include tributary inflows
            # Credit: Glose et al., 2013
            T_L = ((Q_surf/Q_L) * T_q_surf) + ((Q_trib/Q_L) * T_trib) + ((Q_gw/Q_L) * T_q_gw)
        
            
    ## Calibrate Hyporheic Exchange Heat Flux
    # Vary hyporheic flow fraction by stream order
    # Initialize array to store hyporheic flow fractions at each reach
    hyp_frac = np.zeros(n)
    
    # Set hyporheic fraction for 1st, 2nd, and 3rd order streams
    # Hyporheic fraction defined as proportion of streamflow that enters subsurface hyporheic zone
    hyp_frac[stream_order == 1] = hyp_frac1
    hyp_frac[stream_order == 2] = hyp_frac2
    hyp_frac[stream_order == 3] = hyp_frac3
    
    # Initialize array to store hyporheic lagged temperatures
    # Last time step of T_hyp will be empty, last time step isn't used to calculate anything
    T_hyp = np.zeros((n_dt,n))
            
    # Initialize array to store hyporheic flow return rate to stream
    Q_hyp = np.zeros((n_dt,n))
    
    # Set lag time (hours) for hyporheic residence time
    hyp_lag = int(hyp_lag)
    
    ## Solve for future longitudinal temperatures using semi-Lagrangian method ##
    # Approach derived from Yearsley, 2009; Yearsley, 2012
        
    # Set constants
    rho_water = 997 # Units: kg/m3
    c_water= 4182 # Units: J/kg*degC

    ## SEMI-LAGRANGIAN REVERSE PARTICLE TRACKING COMPUTATION SCHEME: DERIVED FROM YEARSLEY, 2009; 2012 ##

    # Set lagrangian model time step
    dt_lg = time_step # minutes

    # Convert time step to seconds
    dt_lg_s = dt_lg * 60 # seconds

    # Determine starting location of parcel at time t 
    # Calculate number of seconds to traverse segment j (upstream of ending node)
    # Integrate velocities at t time step to find starting locations of t+dt nodes

    # Loop through model time steps to calculate temperatures at t+1
    # Calculates temperatures at t+dt using forcings at time t
    for t in range(0,len(mod_time)-1):

        # If lagged time is outside model temporal range:
        if t <= hyp_lag:
            
            # Set hyporheic temperature to AT-GW temperature at t=0 for model spin up period (equal to number of lagged)
            T_hyp[t,:] = T_sim[0,0:n] # Hyporheic temperature based on temperature of upstream node from reach (ignoring outlet node)
            
            # Calculate hyporheic flow return rate by scaling stream discharge by hyporheic fraction at boundary condition (m3/s)
            # Hyporheic flow enters subsurface at rate based on streamflow at time t-hyp_lag
            # Hyporheic flow reenters the stream at at time t at the rate it entered at time t-hyp_lag
            Q_hyp[t,:] = Q[0,:] * hyp_frac
    
        else: # If lagged time is inside model temporal range
        
            # Calculate mean lagged temperature within hyporheic window to represent hyporheic temperature
            T_hyp[t,:] = np.mean(T_sim[t-hyp_lag:t-1,0:n],axis=0)
            
            # Calculate hyporheic flow rate (m3/s) based on mean lagged discharge within hyporheic window
            Q_hyp[t,:] = np.mean(Q[t-hyp_lag:t-1,:],axis=0) * hyp_frac
            
        # Create array to store ending locations of each node point after time step
        x_orig = np.zeros(len(nodes))
        # Create array to store indices of upstream node from origin point
        ind_orig = np.arange(0,len(nodes),1)
        # Create array to store the number of segments traversed in upstream travel
        n_seg = np.arange(0,len(nodes),1)
        # Create array to store interpolated temperatures of origin points
        T_orig = np.zeros(len(nodes))
        
        for j in range(0,len(nodes)):

            ## LAGRANGIAN REVERSE PARTICLE TRACKING ##    
        
            # Initialize t_remain, x_trav, seg_i
            t_remain = dt_lg_s # remaining time in computation step
            x_pos = nodes[j] # x position of water parcel
            seg_i = 1 # segment index
            
            while t_remain > 0:
                
                # Calculate time to traverse upstream segment
                # Traversing from node j to node j-1 passes through node_length[j-seg_i]
                t_up = node_length[j-seg_i] / V_in[t,j-seg_i]
                
                # Check if x_pos <= 0
                if x_pos <= 0:
                    
                    # Set x_end equal to 0: boundary condition
                    x_orig[j] = 0
                    
                    # Set index node location to 0: boundary condition
                    ind_orig[j] = 0
                    
                    # End loop
                    break
                
                # Check if time to traverse segment is greater than remaining time
                if t_remain > t_up:
                        
                    # Find remaining time in time step after traversing segment
                    t_remain = t_remain - t_up
                    
                    # Track location of traveling water parcel
                    # Traversing from node j to node j-1 passes through node_length[j-seg_i]
                    x_pos = x_pos - node_length[j-seg_i]
                    
                    # Index node location
                    ind_orig[j] -= 1
                    
                    # Index segment couter
                    seg_i += 1
                    
                else:
                    
                    # t_remain less than time of next segment: Can't traverse full segment distance
                    # Calculate location within segment based on remaining time and velocity
                    x_lastseg = t_remain * V_in[t,j - seg_i]
                    
                    # Track location of traveling water parcel
                    x_pos = x_pos - x_lastseg
                    
                    # Index node location
                    ind_orig[j] -= 1
                    
                    # Insert ending location in array
                    x_orig[j] = x_pos
                    
                    # End loop
                    break
        
            # Calculate number of nodes traversed by each parcel of water (includes partial final segment)
            # original n_seg value functions as starting reach boundary
            n_seg[j] = n_seg[j] - ind_orig[j]
            
            # Interpolate temperatures between boundaries of upstream/downstream node surrounding origin point
            # Second order lagrangian interpolation using 3 points (Yearsley, 2012)
        
            # If node location is 0 (boundary condition), continue to next loop iteration
            if j == 0:
                
                # Set T equal to boundary condition
                T_orig[j] = T_sim[t,0]
                
            # If starting point is at the boundary condition, set T equal to BC
            if x_orig[j] == 0:
                
                # Set T equal to boundary condition
                T_orig[j] = T_sim[t,0]
        
            # If point upstream of origin node used in interpolation is beyond upstream boundary condition
            if ind_orig[j] == 0:
            
                # Use 3rd node downstream for interpolation instead
                x_lagrange = np.array([nodes[ind_orig[j]], nodes[ind_orig[j]+1], nodes[ind_orig[j]+2]])
                T_lagrange = np.array([T_sim[t,ind_orig[j]], T_sim[t,ind_orig[j]+1], T_sim[t,ind_orig[j]+2]])
                
            else:
            
                # Gather x and T values of 3 nearest node points to origin point (j-1,j,j+1)
                x_lagrange = np.array([nodes[ind_orig[j]-1], nodes[ind_orig[j]], nodes[ind_orig[j]+1]])
                T_lagrange = np.array([T_sim[t,ind_orig[j]-1], T_sim[t,ind_orig[j]], T_sim[t,ind_orig[j]+1]])
        
            # Generate second-order lagrange polynomial
            T_poly = lagrange(x_lagrange, T_lagrange)
            # Evaluate second-order lagrange polynomial to interpolate starting temperature at origin point
            T_orig[j] = T_poly(x_orig[j])


            ## TEMPERATURE CALCULATION ##    
        
            # If parcel traveled X reach segments (not the first node), calculate change in temperature
            # If parcel traveled no reach segments (first node), temperature stays as boundary condition    
            if n_seg[j] > 0:
        
                # Initialize current location of temperature simulation
                x_T = x_orig[j]
        
                # Intialize current temperature simulation
                T_i = T_orig[j]        
        
                # Calculate change in temperature from starting position to next node
                # Loop for number of reaches traveled by parcel of water
                for k in range(0,n_seg[j]): 
                    
                    # Find length of time to traverse current segment
                    dt_i = (nodes[ind_orig[j]+k+1] - x_T)/V_in[t,ind_orig[j]+k] 
                
                    # Calculate radiative forcing using origin temperature along segment (W/m2)
                    R_i = rad_flux(sw_in[t,ind_orig[j]+k],lw_in[t,ind_orig[j]+k],at[t,ind_orig[j]+k],
                                   T_i,spec_h[t,ind_orig[j]+k],wind[t,ind_orig[j]+k],rip_shade[ind_orig[j]+k],
                                   air_press[t,ind_orig[j]+k])[0]
                    
                    # Calculate advective forcing using volume of segment and groundwater inflow (deg/s)
                    phi_i = (Q_L[t,ind_orig[j]+k]/vol_node[t,ind_orig[j]+k]) * (T_L[t,ind_orig[j]+k] - T_i)
                    
                    # Calculate hyporheic forcing (deg/s)
                    # Hyporheic flow based on rate that it entered the stream at lagged time
                    hyp_i = (Q_hyp[t,ind_orig[j]+k]/vol_node[t,ind_orig[j]+k]) * (T_hyp[t,ind_orig[j]+k] - T_i)
                    
                    # Calculate temperature at boundary of current segment
                    T_i = T_i + (dt_i * ((R_i/(rho_water * c_water * chan_depth_mat[t,ind_orig[j]+k])) + phi_i + hyp_i))
                
                    # Increment location of current T_i simulation
                    x_T = nodes[ind_orig[j]+k+1]
                    
                # Add simulated temperature to output matrix    
                T_sim[t+1,j] = T_i
                
    return(T_sim, Q, mod_datetime, nodes)    
    
    
