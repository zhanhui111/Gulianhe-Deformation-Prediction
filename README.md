# Gulianhe Surface Deformation Prediction

Code, derived InSAR observations, and trained models for surface deformation
analysis and prediction at the Gulianhe open-pit coal mine, China.

## Contents

- `src/`: feature construction, GA–PSO–BP training and prediction, and OPGD analysis.
- `data/points_base.csv`: 12,055 analysis points with coordinates, 34-date LOS displacement observations, and terrain attributes.
- `data/orbit76_native_34dates.csv.gz`: complete 34-date LOS displacement time series for 17,255 native orbit-76 points.
- `data/climate_daily.csv`: daily meteorological data; source information is provided in the accompanying metadata.
- `data/orbit149_displacement_rasters/`: 59 orbit-149 LOS displacement rasters in ENVI format, with `.hdr` and `.sml` files. Displacement units are millimetres. Keep each raster and its accompanying files together.
- `results/anchored_bp/`: trained five-network model, selected settings, and prediction metrics.

Prediction and OPGD analyses use orbit-76 observations. The orbit-149 rasters
provide complementary observations and are not required to run the prediction model.

## Run

Python 3.12 was used. Install the dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Reproduce the main model's prediction metrics using the saved weights:

```bash
python -B reproduce.py
```

Retrain the main model with the saved settings and evaluate its predictions:

```bash
python -B reproduce.py --train
```

Outputs are written to `generated/`; the supplied model weights are retained.
The model uses training targets through 1 November 2020 and is evaluated on
2021 observations.

## Main results

| Forecast lead (days) | R² | RMSE (mm) | MAE (mm) |
|---|---:|---:|---:|
| 36–72 | 0.87 | 11.53 | 8.29 |
| 60–84 | 0.83 | 13.31 | 9.51 |
| 96–132 | 0.77 | 15.59 | 11.36 |

Errors are measured against the InSAR LOS displacement observations.
