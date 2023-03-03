# Visualization

This folder contains two scripts used to create a set of figures used to interpret the performance of the water temperature model.

-   **nwm_st_visualization.py**: A script that acts as the base for figure creation, creating plots related to error metrics from each model configuration.
-   **gw_temp_visualization**: A script used to create figures related to time series of estimated groundwater temperatures.

## Figures

We provide both final figure PNGs and the Adobe Illustrator files used to modify raw figures exported from Matplotlib or ArcGIS.

-   **./figures**: Folder containing the following figures:

    -   **./figure1**: Site map of model river reaches and water temperature gages in H.J. Andrews Experimental Forest, OR, USA.

    -   **./figure2**: Modeled atmospheric, radiative, and hydrologi cheat fluxes and associated National Water Model input data.

    -   **./figure3**: Estimated groundwater inflow temperature susing tuned source depth approach.

    -   **./figure4**: Simulated water temperature RMSE across model calibration runs.

    -   **./figure5**: Comparison of model configuration performance across three metrics of error.

    -   **./figure6**: Envelopes of well-calibrated water temperature simulations at headwater gage.

    -   **./figure7**: Envelopes of well-calibrated water temperature simulations at the outlet gage.

    -   **./model_formulation**: Conceptual depiction of water temperature modeling framework

    -   **./model_calibration**: Conceptual depiction of model configurations (M1-M4)

## Tables

-   **./tables**: Folder containing the following tables:

    -   **./table1**: Summary of water temperature model formulations.

    -   **./table2**: Parameter definitions and tuning ranges for model configurations.

    -   **./table3**: Optimal parameter values for best model calibration runs.
