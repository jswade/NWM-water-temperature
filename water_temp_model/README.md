# Water Temperature Model

-   **nwm_st_model_base.py**: Main model function used to calculate heat fluxes and simulate water temperatures along NWM reaches.
-   **nwm_st_model_runs.py**: Script used to perform model calibration, which calls the **nwm_st_model_base.py** function.

## Model Description
The water temperature model presented here is a semi-Lagrangian advection model based on the work of Yearsley, 2009. The model functions by quantifying heat fluxes into and out of the stream using National Water Model data and calculating the resultant changes in water temperature in relation to an upstream boundary condition. Modeled heat fluxes include solar radiation, longwave radiation, latent heat, sensible heat, groundwater inflow, surface runoff, and hyporheic exchange. The model simulates temperatures using 1-km reach segments coincident with National Water Model/NHD reaches at an hourly time step within the H.J. Andrews test catchment. Upstream boundary conditions of each tributary was equal to the temperature of groundwater, which was estimated using a calibrated approach that buffered and lagged daily mean air temperatures.

<img src="https://github.com/jswade/NWM-water-temperature/blob/main/visualization/figures/model_formulation/model_formulation.png" width="70%" height="70%">

## Model Calibration
Four model configurations of sequentially increasing complexity (M1, M2, M3, M4) were tested to evaluate tradeoffs between performance and efficiency. Progressing from M1 to M4, the model represents additional processes by adding degrees of freedom in calibration.
<br/>
<br/>
The four model configurations were tuned using uniform Monte Carlo sampling of uncertain parameters evaluated across 5000 model runs of each configuration. From these 5000 runs, the top 1% of model runs (50 runs) ranked by a weighted error of RMSE at the headwater and outlet gage were used to assess the peak potential performance of each configuration.

<img src="https://github.com/jswade/NWM-water-temperature/blob/main/visualization/figures/model_calibration/model_calibration.png" width="70%" height="70%">

## References
Yearsley, J. R. (2009). A semi-Lagrangian water temperature model for advection-dominated river systems. Water Resources Research, 45(12). https://doi.org/10.1029/2008WR007629
