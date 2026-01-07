# WIDS-5.0-Solar-site-suitability
## Solar Site Suitability Smart Mapping using Google Earth Engine

This project implements a **physics-aware, regression-based approach** to solar site suitability mapping using satellite and reanalysis data in **Google Earth Engine (GEE)**.

Instead of directly classifying locations based on existing solar installations, the project models **Global Horizontal Irradiance (GHI)** as a continuous physical variable and derives solar suitability from it.

---

## Study Region
- **Region:** Gujarat, India
- **Selection Rationale:** High solar deployment density and diverse terrain and land-cover conditions

Administrative boundaries were used to define the study region.
The data of solar power plants was taken from a public dataset named **[global-databse-power plants](https://www.wri.org/research/global-database-power-plants)** by World Resources Institute(WRI). Then the solar plants of India were filtered out from the dataset and that was uploaded to the Google Earth Engine. We then defined the study region which contained only the solar plower plants present in the state of Gujarat. 

---

## Objective
To learn the relationship between terrain, surface, and atmospheric features and **solar energy availability**, and generate a **continuous solar suitability map**.

---

## Target Variable (Regression Output)
- **Global Horizontal Irradiance (GHI)**
- Dataset: `ERA5-Land Daily Aggregated`
- Band used: `surface_solar_radiation_downwards_sum`
- Temporal aggregation: Annual mean (2020)

---

## Input Features
The following satellite-derived features were used as predictors:

| Feature | Dataset |
|------|--------|
| Elevation | SRTM DEM |
| Slope | Derived from DEM |
| Aspect | Derived from DEM |
| Land Cover | MODIS MCD12Q1 |
| Cloud proxy | ERA5-Land net solar radiation |
| Surface Albedo | MODIS MCD43A3 |

---

## Machine Learning Model
- **Model:** Random Forest Regressor
- **Training platform:** Google Earth Engine
- **Sampling:** Pixel-wise sampling (1 km resolution)
- **Output:** Continuous predicted GHI surface

---

## Solar Suitability Index
The predicted GHI surface was normalized to produce a **Solar Suitability Index (0–1)**:
- 0 → Low suitability
- 1 → High suitability

This allows post-hoc classification into high / medium / low suitability zones.

---

## Results (Upto Week 3)
- Generated a continuous **predicted GHI map** over Gujarat
- Produced a **solar suitability index** highlighting physically favorable locations
- Avoided deployment-history bias inherent in classification-only approaches

---

## 🔗 Google Earth Engine Script
You can view and run the full GEE script here, the code is present in the file named studyRegion_Gujarat.:

**[GEE Script Link](https://code.earthengine.google.co.in/eac47770f097ff2805e69ace9f4a9abb?accept_repo=users%2Fujavalgandhi%2FEnd-to-End-GEE)**

---

## Notes
- Existing solar plant locations were used for spatial understanding and validation, not as training labels.
- This approach is transferable to other regions without retraining.

---

## Future Work
- Validation against ground-measured solar data
