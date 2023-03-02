# NWM Channels

-   **nwm_channel_download.py**: Downloads and formats NWM channel channel data located within H.J. Andrews Experimental Forest.
-   **/raw_data**: Data used to derive channel locations and parameters, including:
    -   **/NWM_parameters_v2.1**: NWM parameter files related to gridded and river channel features. (Source: https://www.nohrsc.noaa.gov/pub/staff/keicher/NWM_live/NWM_parameters/NWM_parameter_files.tar.gz)
    -   **/NWM_v2.1_channel_hydrofabric**: Geometry objects defining the location and identity of NWM reaches. (Source: https://www.nohrsc.noaa.gov/pub/staff/keicher/NWM_live/web/data_tools/NWM_channel_hydrofabric.tar.gz)
    -   **/reach_formatting**: Locations of observed water temperature gages along NWM reaches.
    -   **/shapefiles**: Shapefile of the H.J. Andrews test catchment.
-   **/formatted_channels**: Formatted channel parameters and identification within H.J. Andrews.
    -   **Note**" In order to facilitate tributary connections, the model was run in three independent cycles. We manually assigned reaches into three groups based on their relative position within the stream network (see **hja_channels_diagram.png**). These groups included the first model cycle (reaches Fa, Fb, Fc, Fd), the second model cycle (reach Sa), and the mainstem model cycle (reach Ma).
