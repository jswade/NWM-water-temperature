# Energy balance modeling of river water temperatures within the National Water Model Framework.

Abstract/Intro

## Contents

-   **./nwm_data_download**: Download and extract NWM v2.1 Retrospective data at H.J. Andrews Experimental Forest study catchment


-   **./data_raw**: 

-   **./data_results**: 

-   **./metric_calculation**: 

-   **./model_runs**: 

-   **./rf_model_tuning**: 

-   **./site_classification**: 

-   **./site_id**: 

-   **./spearman**: 

-   **./visualization**: 


Model Environment/YAML?

## Procedure

All code is written in R. File references are relative to the current repository and should be changed to match your local directory. Additional information on each step of our analysis is available within individual repository folders.

To recreate our analysis, heed the following instructions:

1.  Run **GAGES_StreamTemp_Sites.R** in **./site_id** to identify sites used in the analysis.

2.  Within **./data_download**, run the following scripts:

    -   **./AirTemperature**: **PRISM_data_formatting.R** to extract daily air temperature at each site.

    -   **./Discharge**: **Discharge_DataDownload.R** to download daily USGS discharge data at each site.

    -   **./StreamTemperature**: **ST_DataDownload.R** to download daily USGS ST data at each site.

3.  Run **max_calc.R** and **thermal_sensitivity_calc.R** in **./metric_calculation** to calculate ST metrics. *(Note: Thermal Sensitivity and Slope are used interchangeably in this repository)*

4.  View **./StratifiedSampling** in **./site_classification** to confirm manual-determined stratified sampling groups.

5.  Run **RF_runs_hyperparameter_tuning.R** in **./rf_model_tuning** to optimize *mtry* in RF models.

6.  Run **RF_runs.R and RF_pred_runs.R** *in* **./model_runs** to fit monthly RF models to different combinations of metrics and sites.

7.  Perform Spearman's rank correlation between predictors and metrics using **RF_spearman.R** scripts in **./spearman**.

8.  Generate final figures using **RF_vis_run.R** in **./visualization**. *(Note: Many figures were altered for publication using Adobe Illustrator. Where applicable, Illustrator files are included.)*
