// ==============================================================================
// Solar Site Suitability Smart Mapping using Google Earth Engine (GEE)
// Study Region: Gujarat, India
// Target: Annual Mean GHI (ERA5-Land 2020)
// Features: SRTM Elevation, Slope, Aspect, MODIS Land Cover, Cloud Proxy, Albedo
// Model: Random Forest Regressor (ee.Classifier.smileRandomForest)
// ==============================================================================

// 1. Study Region: Select Gujarat
var gujarat = states.filter(ee.Filter.eq('ADM1_NAME', 'Gujarat'));

Map.addLayer(
  gujarat.style({
    color: '#08306b',
    fillColor: '00000000',
    width: 2
  }),
  {},
  'Gujarat Boundary'
);

Map.centerObject(gujarat, 6);

// 2. Build geometry from lat/lon for Ground Truth Solar Plants
var solarPoints = solarPoints3.map(function(f){
  var lon = ee.Number(f.get('lon'));
  var lat = ee.Number(f.get('lat'));
  return ee.Feature(ee.Geometry.Point([lon, lat]), f.toDictionary());
});

// Filter solar plants to Gujarat
var solarGujarat = solarPoints.filterBounds(gujarat);

// Visualize Gujarat solar plants
Map.addLayer(
  solarGujarat.style({
    color: '#d7191c', // strong red
    pointSize: 5
  }),
  {},
  'Solar Plants in Gujarat (Validation)'
);

// 3. Target Variable: Annual Mean GHI (Global Horizontal Irradiance)
var ghi = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
  .select("surface_solar_radiation_downwards_sum")
  .filterDate("2020-01-01", "2020-12-31")
  .mean()
  .clip(gujarat)
  .rename("MEAN_GHI");

Map.addLayer(
  ghi,
  {min: 1e7, max: 2.5e7, palette: ['blue', 'pink', 'red'], opacity: 0.6}, 
  "GHI Target (ERA5-Land)"
);

// 4. Input Features
// 4.1 Elevation
var dem = ee.Image("USGS/SRTMGL1_003")
  .clip(gujarat)
  .rename("elevation");
  
// 4.2 Slope and Aspect
var terrain = ee.Terrain.products(dem);
var slope = terrain.select("slope").rename("slope");
var aspect = terrain.select("aspect").rename("aspect");

// 4.3 Land Cover
var lulc = ee.Image("MODIS/006/MCD12Q1/2019_01_01")
  .select("LC_Type1")
  .clip(gujarat)
  .rename("landcover");
  
// 4.4 Cloud Fraction / Net Solar Radiation Proxy
var cloud = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
  .select("surface_net_solar_radiation_sum")
  .filterDate("2020-01-01", "2020-12-31")
  .mean()
  .clip(gujarat)
  .rename("cloud_proxy");
  
// 4.5 Surface Albedo
var albedo = ee.ImageCollection("MODIS/061/MCD43A3")
  .select("Albedo_WSA_shortwave")
  .filterDate("2020-01-01", "2020-12-31")
  .mean()
  .clip(gujarat)
  .rename("albedo");

// 5. Creating Multi-Band Feature Stack
var featureStack = ee.Image.cat([
  dem,
  slope,
  aspect,
  lulc,
  cloud,
  albedo,
  ghi
]);

var featureNames = [
  "elevation",
  "slope",
  "aspect",
  "landcover",
  "cloud_proxy",
  "albedo"
];

// ==============================================================================
// 6. Sampling & Statistical Train / Validation Split
// ==============================================================================
// Sample 4,000 pixels across Gujarat at 1 km resolution
// geometries: true enables coordinate extraction for spatial cross-validation
var allSamples = featureStack.sample({
  region: gujarat,
  scale: 1000,
  numPixels: 4000,
  seed: 42,
  geometries: true
});

// 6.1 Random Train/Test Split (80% Train, 20% Validation)
var withRandom = allSamples.randomColumn('random', 42);
var trainSet = withRandom.filter(ee.Filter.lt('random', 0.8));
var testSet = withRandom.filter(ee.Filter.gte('random', 0.8));

print('Total Samples:', allSamples.size());
print('Train Samples (80%):', trainSet.size());
print('Validation Samples (20%):', testSet.size());

// 7. Training Random Forest Regressor
var regressor = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  seed: 42
}).setOutputMode('REGRESSION');

var trainedModel = regressor.train({
  features: trainSet,
  classProperty: "MEAN_GHI",
  inputProperties: featureNames
});

// ==============================================================================
// 8. Quantitative Evaluation on Test Set (R², RMSE, MAE, Feature Importance)
// ==============================================================================
var validated = testSet.classify(trainedModel, 'predicted_GHI');

// Residual calculation: error = actual - predicted
var withResiduals = validated.map(function(feat) {
  var actual = ee.Number(feat.get('MEAN_GHI'));
  var predicted = ee.Number(feat.get('predicted_GHI'));
  var diff = actual.subtract(predicted);
  return feat.set({
    'diff': diff,
    'sq_diff': diff.pow(2),
    'abs_diff': diff.abs()
  });
});

// Compute Pearson correlation (r) -> R²
var correlation = ee.Number(
  withResiduals.reduceColumns(ee.Reducer.pearsonsCorrelation(), ['MEAN_GHI', 'predicted_GHI'])
  .get('correlation')
);
var r2 = correlation.pow(2);

// Compute RMSE
var meanSqDiff = ee.Number(
  withResiduals.reduceColumns(ee.Reducer.mean(), ['sq_diff'])
  .get('mean')
);
var rmse = meanSqDiff.sqrt();

// Compute MAE
var mae = ee.Number(
  withResiduals.reduceColumns(ee.Reducer.mean(), ['abs_diff'])
  .get('mean')
);

// Target mean and Relative RMSE
var actualMean = ee.Number(
  withResiduals.reduceColumns(ee.Reducer.mean(), ['MEAN_GHI'])
  .get('mean')
);
var rrmsePercent = rmse.divide(actualMean).multiply(100);

print('================ MODEL EVALUATION METRICS ================');
print('Validation R² (Coefficient of Determination):', r2);
print('Validation RMSE (J/m²/day):', rmse);
print('Validation MAE (J/m²/day):', mae);
print('Relative RMSE (%):', rrmsePercent);

// Feature Importance
var explanation = trainedModel.explain();
print('Random Forest Feature Importance:', explanation.get('importance'));

// ==============================================================================
// 9. Spatial Block Holdout Split (Testing Spatial Generalization)
// ==============================================================================
// Interview Defense: In spatial modeling, random splits can leak variance via
// Tobler's First Law (spatial autocorrelation). We test spatial holdout by
// holding out Eastern Gujarat (> 72.5°E longitude) as unseen geographic test block.
var withCoords = allSamples.map(function(feat) {
  var coords = feat.geometry().coordinates();
  return feat.set('lon', coords.get(0)).set('lat', coords.get(1));
});

var spatialTrain = withCoords.filter(ee.Filter.lte('lon', 72.5));
var spatialTest  = withCoords.filter(ee.Filter.gt('lon', 72.5));

var spatialModel = ee.Classifier.smileRandomForest({
  numberOfTrees: 100,
  seed: 42
}).setOutputMode('REGRESSION').train({
  features: spatialTrain,
  classProperty: "MEAN_GHI",
  inputProperties: featureNames
});

var spatialValidated = spatialTest.classify(spatialModel, 'spatial_pred');
var spatialCorr = ee.Number(
  spatialValidated.reduceColumns(ee.Reducer.pearsonsCorrelation(), ['MEAN_GHI', 'spatial_pred'])
  .get('correlation')
);
print('Spatial Holdout Validation R² (East Gujarat block):', spatialCorr.pow(2));

// ==============================================================================
// 10. Generating Continuous Predicted GHI Surface & Solar Suitability Index
// ==============================================================================
var predictedGHI = featureStack
  .select(featureNames)
  .classify(trainedModel)
  .rename("Predicted_GHI");

Map.addLayer(
  predictedGHI,
  {
    min: 1.3e7,
    max: 2.4e7,
    palette: ['#4575b4', '#fee090', '#d73027'],
    opacity: 0.8
  },
  'Predicted GHI'
);

// Converting regression output to Solar Suitability Index (unitScale 0 to 1)
var suitability = predictedGHI
  .unitScale(1e7, 2.5e7)
  .rename("Suitability");

Map.addLayer(
  suitability,
  {
    min: 0,
    max: 1,
    palette: ['#2c7bb6', '#ffffbf', '#d7191c'],
    opacity: 0.85
  },
  'Solar Suitability Index'
);

// 11. Export Final GeoTIFF
Export.image.toDrive({
  image: suitability,
  description: 'Solar_Suitability_Gujarat',
  region: gujarat.geometry(),
  scale: 1000,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
