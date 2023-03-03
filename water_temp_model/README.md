# Water Temperature Model

-   **nwm_st_model_base.py**: Main model function used to calculate heat fluxes and simulate water temperatures along NWM reaches.
-   **nwm_st_model_runs.py**: Script used to perform model calibration, calling **nwm_st_model_base.py** function.

## Model Description
The water temperature model presented here is a semi-Lagrangian advection model based on the work of Yearsley, 2009. The model functions by quantifying heat fluxes into and out of the stream using National Water Model data and calculating the resultant changes in water temperature in relation to an upstream boundary condition. Modeled heat fluxes include solar radiation, longwave radiation, latent heat, sensible heat, groundwater inflow, surface runoff, and hyporheic exchange. The model simulates temperatures using 1-km reach segments coincident with National Water Model/NHD reaches at an hourly time step. Upstream boundary conditions of each tributary were set to equal the temperature of groundwater, which was estimated using a calibrated approach that buffered and lagged daily mean air temperatures.

<img src="visualization/figures/model_formulation/model_formulation.png" width="70%" height="70%">

## Model Calibration
Four model configurations of sequentially increasing complexity (M1, M2, M3, M4) were tested to evaluate tradeoffs between performance and efficiency. These configurations were defined by the set of parameters they tuned:

<img src="visualization/figures/model_calibration/model_calibration.png" width="70%" height="70%">

