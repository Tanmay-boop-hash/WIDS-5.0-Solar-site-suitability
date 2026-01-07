
// Select Gujarat
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

// Build geometry from lat/lon
var solarPoints = solarPoints3.map(function(f){
  var lon = ee.Number(f.get('lon'));
  var lat = ee.Number(f.get('lat'));
  return ee.Feature(ee.Geometry.Point([lon, lat]), f.toDictionary());
});

// Filter solar plants to Gujarat
var solarGujarat = solarPoints.filterBounds(gujarat);

// Visualize Gujarat solar plants
Map.addLayer(
  solarGujarat.style({color: '#d7191c', // strong red
  pointSize: 5}),
  {},
  'Solar Plants in Gujarat'
);

// GHI(Global Horizontal Irradiance)
var ghi = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
  .select("surface_solar_radiation_downwards_sum")
  .filterDate("2020-01-01", "2020-12-31")
  .mean()
  .clip(gujarat)
  .rename("MEAN_GHI");

Map.addLayer(
  ghi,
  {min: 1e7, max: 2.5e7, palette: ['blue','pink','red'], opacity: 0.6}, 
  "GHI (Target)"
);

// Loading elevation
var dem = ee.Image("USGS/SRTMGL1_003")
  .clip(gujarat)
  .rename("elevation");
  
// Slope and Aspect
var terrain = ee.Terrain.products(dem);
var slope = terrain.select("slope").rename("slope");
var aspect = terrain.select("aspect").rename("aspect");

// Land cover
var lulc = ee.Image("MODIS/006/MCD12Q1/2019_01_01")
  .select("LC_Type1")
  .clip(gujarat)
  .rename("landcover");
  
// Cloud Fraction
var cloud = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
  .select("surface_net_solar_radiation_sum")
  .filterDate("2020-01-01", "2020-12-31")
  .mean()
  .clip(gujarat)
  .rename("cloud_proxy");
  
// Surface Albedo
var albedo = ee.ImageCollection("MODIS/061/MCD43A3")
  .select("Albedo_WSA_shortwave")
  .filterDate("2020-01-01", "2020-12-31")
  .mean()
  .clip(gujarat)
  .rename("albedo");

// Creating Feature Stack
var featureStack = ee.Image.cat([
  dem,
  slope,
  aspect,
  lulc,
  cloud,
  albedo,
  ghi
]);
// each pixel now looks like: [elevation, slope, aspect, landcover, cloud, albedo] → GHI


// sample pixels for training the model
var samples = featureStack.sample({
  region: gujarat,
  scale: 1000,          // 1 km resolution (FAST + reasonable)
  numPixels: 4000,      // Enough for regression
  seed: 42,
  geometries: false
});

// Training Regression model(Random Forest)
var regressor = ee.Classifier.smileRandomForest({
  numberOfTrees: 100
}).setOutputMode('REGRESSION');

var trainedModel = regressor.train({
  features: samples,
  classProperty: "MEAN_GHI",
  inputProperties: [
    "elevation",
    "slope",
    "aspect",
    "landcover",
    "cloud_proxy",
    "albedo"
  ]
});

// Applying the model
var predictedGHI = featureStack
  .select([
    "elevation",
    "slope",
    "aspect",
    "landcover",
    "cloud_proxy",
    "albedo"
  ])
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


// Converting the output of regression model to solar suitability 
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








