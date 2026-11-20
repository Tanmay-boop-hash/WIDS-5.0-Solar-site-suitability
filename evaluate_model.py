"""
Solar Site Suitability Regression - Local Evaluation & Benchmark Suite
Study Area: Gujarat, India
Target: Global Horizontal Irradiance (GHI) in J/m²/day
Predictors: Elevation, Slope, Aspect, Land Cover, Cloud Proxy, Surface Albedo

This script allows reproducible, offline evaluation of the Random Forest
regression model on pixel samples representative of the Gujarat study area.
It computes R², RMSE, MAE, feature importances, and spatial cross-validation.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def generate_representative_dataset(n_samples=4000, random_state=42):
    """
    Generates realistic pixel samples matching Gujarat geography and ERA5-Land/SRTM distributions.
    Lon: 68.2 to 74.5, Lat: 20.1 to 24.7
    """
    rng = np.random.RandomState(random_state)
    
    # Gujarat spatial extent
    lon = rng.uniform(68.5, 74.2, n_samples)
    lat = rng.uniform(20.3, 24.5, n_samples)
    
    # Elevation: High in Saurashtra/Gir & Eastern highlands (0 - 800m), low coastal plains
    base_elev = 20 + 200 * (lon - 68.5) / 5.7 + 100 * np.sin(lat * 3)
    elevation = np.clip(base_elev + rng.exponential(scale=60, size=n_samples), 0, 1100)
    
    # Slope and aspect derived from topography
    slope = np.clip(rng.gamma(shape=1.5, scale=2.5, size=n_samples) * (elevation / 100 + 0.5), 0, 45)
    aspect = rng.uniform(0, 360, n_samples)
    
    # Land Cover classes (MODIS IGBP: 1-17, Gujarat dominated by Cropland, Grassland, Barren, Urban)
    landcover_classes = [7, 8, 9, 10, 12, 13, 14, 16] # Grass, Shrub, Savanna, Cropland, Barren
    landcover = rng.choice(landcover_classes, size=n_samples, p=[0.1, 0.15, 0.1, 0.35, 0.05, 0.05, 0.1, 0.1])
    
    # Surface Albedo (0.12 - 0.35, higher in Rann of Kutch & arid barren zones)
    is_kutch = (lat > 23.0) & (lon < 71.5)
    albedo = np.where(is_kutch, rng.normal(0.28, 0.04, n_samples), rng.normal(0.18, 0.03, n_samples))
    albedo = np.clip(albedo, 0.08, 0.45)
    
    # Cloud Proxy (surface net solar radiation, ERA5-Land, ~ 1.0e7 - 1.8e7 J/m²)
    # South Gujarat receives more precipitation/cloud cover; North/West (Kutch) has clearer skies
    cloud_proxy = 1.65e7 - (lat - 20.0) * 0.03e7 - (74.0 - lon) * 0.04e7 + rng.normal(0, 0.08e7, n_samples)
    cloud_proxy = np.clip(cloud_proxy, 1.1e7, 1.95e7)
    
    # Physical formulation for Annual Mean GHI (ERA5-Land surface_solar_radiation_downwards_sum)
    # Gujarat GHI range: ~1.7e7 to 2.3e7 J/m²/day (~4.7 to 6.4 kWh/m²/day)
    # Influenced strongly by atmospheric transmittance (cloud proxy), latitude, elevation, albedo feedback
    true_ghi = (
        0.88 * cloud_proxy
        + 1.8e6 * ((lat - 20.0) / 4.5)           # Latitudinal solar geometry
        + 1200 * elevation                       # Elevation optical path thinning
        - 8000 * slope                           # Slope shadowing effect
        + 1.2e6 * (albedo - 0.18)                # Backscatter
        + 0.5e6 * (np.cos(np.radians(aspect - 180))) # South-facing aspect benefit
        + rng.normal(0, 3.5e5, n_samples)        # Atmospheric/measurement noise
    )
    
    df = pd.DataFrame({
        'lon': lon,
        'lat': lat,
        'elevation': elevation,
        'slope': slope,
        'aspect': aspect,
        'landcover': landcover,
        'cloud_proxy': cloud_proxy,
        'albedo': albedo,
        'MEAN_GHI': true_ghi
    })
    
    return df


def evaluate_model(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("SOLAR SITE SUITABILITY REGRESSION EVALUATION")
    print("Study Area: Gujarat, India | Target: ERA5-Land Annual Mean GHI")
    print("=" * 65)
    
    # 1. Dataset Generation & Loading
    df = generate_representative_dataset(n_samples=4000, random_state=42)
    feature_cols = ['elevation', 'slope', 'aspect', 'landcover', 'cloud_proxy', 'albedo']
    target_col = 'MEAN_GHI'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. Standard Random Train-Test Split (80/20)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42
    )
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Calculate Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mean_val = np.mean(y_test)
    rrmse_pct = (rmse / mean_val) * 100
    
    # In kWh/m²/day (1 kWh = 3.6e6 J)
    rmse_kwh = rmse / 3.6e6
    mae_kwh = mae / 3.6e6
    
    print("\n--- 1. RANDOM SPLIT VALIDATION (80% Train, 20% Test) ---")
    print(f"Total Samples           : {len(df)}")
    print(f"Training Samples        : {len(X_train)}")
    print(f"Validation Samples      : {len(X_test)}")
    print(f"R² Score                : {r2:.4f}")
    print(f"RMSE                    : {rmse:,.0f} J/m²/day ({rmse_kwh:.3f} kWh/m²/day)")
    print(f"MAE                     : {mae:,.0f} J/m²/day ({mae_kwh:.3f} kWh/m²/day)")
    print(f"Relative RMSE (RRMSE)   : {rrmse_pct:.2f}%")
    
    # 3. Spatial Cross-Validation (Longitudinal Block Split: East vs West Gujarat)
    # Addressing Tobler's First Law and spatial autocorrelation in interviews
    print("\n--- 2. SPATIAL BLOCK CROSS-VALIDATION (Spatial Holdout) ---")
    spatial_threshold = 72.0  # Split Gujarat at 72.0°E (West: Saurashtra/Kutch, East: Mainland)
    west_mask = df['lon'] <= spatial_threshold
    east_mask = df['lon'] > spatial_threshold
    
    rf_spatial = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_spatial.fit(X[west_mask], y[west_mask])
    y_east_pred = rf_spatial.predict(X[east_mask])
    
    spatial_r2 = r2_score(y[east_mask], y_east_pred)
    spatial_rmse = np.sqrt(mean_squared_error(y[east_mask], y_east_pred))
    print(f"Spatial Train (West Gujarat <= 72°E): {west_mask.sum()} samples")
    print(f"Spatial Test  (East Gujarat > 72°E) : {east_mask.sum()} samples")
    print(f"Spatial Holdout R²                  : {spatial_r2:.4f}")
    print(f"Spatial Holdout RMSE                : {spatial_rmse:,.0f} J/m²/day ({spatial_rmse/3.6e6:.3f} kWh/m²/day)")
    
    # 4. Feature Importance
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("\n--- 3. FEATURE IMPORTANCE (Gini Impurity Reduction) ---")
    for feat, imp in feat_imp.items():
        print(f"  {feat:<15}: {imp * 100:.2f}%")
        
    # 5. Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.3)
    
    # Plot A: Actual vs Predicted Scatter
    ax1 = axes[0]
    ax1.scatter(y_test / 3.6e6, y_pred / 3.6e6, alpha=0.4, color='#1f77b4', edgecolors='none', s=25)
    lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]), max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
    ax1.plot(lims, lims, 'r--', lw=2, label='1:1 Ideal Fit')
    ax1.set_xlabel('Actual GHI (kWh/m²/day)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted GHI (kWh/m²/day)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Actual vs. Predicted GHI\n$R^2$ = {r2:.3f}, RMSE = {rmse_kwh:.3f} kWh/m²', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # Plot B: Residual Distribution
    ax2 = axes[1]
    residuals = (y_test - y_pred) / 3.6e6
    ax2.hist(residuals, bins=30, color='#2ca02c', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('Residual Error (kWh/m²/day)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Pixel Count', fontsize=11, fontweight='bold')
    ax2.set_title(f'Residual Error Distribution\nMean = {residuals.mean():.4f}, Std = {residuals.std():.3f}', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Plot C: Feature Importance Bar Chart
    ax3 = axes[2]
    feat_imp.sort_values(ascending=True).plot(kind='barh', ax=ax3, color='#ff7f0e', edgecolor='black', alpha=0.85)
    ax3.set_xlabel('Relative Importance (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Random Forest Feature Importance', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    eval_plot_path = os.path.join(output_dir, 'model_evaluation_metrics.png')
    plt.tight_layout()
    plt.savefig(eval_plot_path, dpi=300)
    plt.close()
    print(f"\n[Saved evaluation figure to: {eval_plot_path}]")
    
    # Save CSV metrics for easy reference
    summary_df = pd.DataFrame({
        'Metric': ['Validation R²', 'Validation RMSE (J/m²)', 'Validation RMSE (kWh/m²)',
                   'Validation MAE (kWh/m²)', 'Relative RMSE (%)', 'Spatial Holdout R²'],
        'Value': [f"{r2:.4f}", f"{rmse:,.0f}", f"{rmse_kwh:.3f}", f"{mae_kwh:.3f}", f"{rrmse_pct:.2f}%", f"{spatial_r2:.4f}"]
    })
    csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"[Saved summary metrics table to: {csv_path}]")
    print("=" * 65)
    return r2, rmse, mae, spatial_r2


if __name__ == '__main__':
    evaluate_model()
