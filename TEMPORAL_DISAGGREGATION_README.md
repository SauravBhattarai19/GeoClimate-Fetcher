# ⏰ Temporal Disaggregation Tool

## Overview

The Temporal Disaggregation tool is designed for high-resolution precipitation analysis, allowing researchers to compare daily station precipitation data with sub-daily satellite products, apply bias correction, and perform temporal disaggregation to 30-minute resolution.

## Features

- **Sub-daily Analysis**: Work with hourly and 30-minute satellite precipitation data
- **Fixed Datasets**: Pre-configured with ERA5-Land (hourly), GPM IMERG (30-min), and GSMaP (hourly)
- **Bias Correction**: Apply systematic bias correction using ground observations
- **Optimal Selection**: Weighted scoring system to select the best satellite product
- **Temporal Disaggregation**: Convert hourly data to 30-minute resolution
- **Corrected Data Export**: Download bias-corrected high-resolution datasets

## Pre-configured Satellite Datasets

### 1. ERA5-Land Hourly
- **Earth Engine ID**: `ECMWF/ERA5_LAND/HOURLY`
- **Band**: `total_precipitation`
- **Resolution**: Hourly
- **Coverage**: 1950-2025
- **Units**: meters (total accumulation, converted to mm)
- **Type**: Total values

### 2. GPM IMERG V07
- **Earth Engine ID**: `NASA/GPM_L3/IMERG_V07`
- **Band**: `precipitation`
- **Resolution**: 30 minutes
- **Coverage**: 2000-2025
- **Units**: mm/hr (rate values, converted to daily mm)
- **Type**: Rate values

### 3. GSMaP Operational V8
- **Earth Engine ID**: `JAXA/GPM_L3/GSMaP/v8/operational`
- **Band**: `hourlyPrecipRate`
- **Resolution**: Hourly
- **Coverage**: 1998-2025
- **Units**: mm/hr (rate values, converted to daily mm)
- **Type**: Rate values

## Required Data Formats

### Daily Precipitation Data (CSV)
```csv
Date,STATION_001,STATION_002,STATION_003
2020-01-01,0.0,0.0,0.0
2020-01-02,5.2,4.8,5.7
2020-01-03,0.0,0.0,0.0
```

**Requirements:**
- Date format: YYYY-MM-DD
- Station columns: Each station should have its own column with station ID as header
- Precipitation units: mm/day
- Missing values: use 0.0
- Station IDs must match those in metadata file

### Station Metadata (CSV)
```csv
id,lat,long,start_date,end_date
STATION_001,40.123,-75.456,2010-01-01,2023-12-31
STATION_002,40.234,-75.567,2015-01-01,2023-12-31
```

**Requirements:**
- Coordinates: decimal degrees (WGS84)
- Date format: YYYY-MM-DD
- Station IDs must be unique

## Workflow

### Step 1: Data Upload
1. Upload daily precipitation data CSV file
2. Upload station metadata CSV file
3. Data validation and preview

### Step 2: Time Range Selection
1. Tool auto-detects available data period
2. User selects analysis start and end dates
3. Validation of temporal coverage

### Step 3: Satellite Data Download
1. Automatic download of all three satellite datasets
2. Point extraction at each station location
3. Progress tracking for multiple stations

### Step 4: Analysis and Results
1. Aggregate satellite data to daily totals
2. Calculate statistical metrics for each dataset
3. Apply composite scoring for optimal selection
4. Perform bias correction
5. Generate corrected high-resolution datasets

## Statistical Methods

### Composite Score Calculation
The optimal dataset is selected using a weighted composite score:

```
Score = 2/(1+RMSE) + 2×Correlation + 1/(1+|Bias|) + 1.5×KGE
```

Where:
- **RMSE**: Root Mean Square Error
- **Correlation**: Pearson correlation coefficient
- **Bias**: Mean difference (satellite - station)
- **KGE**: Kling-Gupta Efficiency

### Bias Correction
Systematic bias correction using regression through origin:

```
P_corrected = α × P_satellite
```

Where α is the scaling factor calculated as:
- For ≥5 data pairs: `α = Σ(station×satellite) / Σ(satellite²)`
- For <5 data pairs: `α = Σ(station) / Σ(satellite)`
- Constrained between 0.1 and 10.0

### Temporal Disaggregation Methods

#### IMERG-Guided Pattern-Based Disaggregation
When IMERG 30-minute data is available for the same period, its sub-hourly distribution pattern is used to disaggregate hourly data:

```
P_30min,t = P_hour × (P_IMERG,t / Σ_hour P_IMERG,t)
```

Where P_30min,t is the disaggregated precipitation for a specific 30-minute interval, P_hour is the hourly value from the corrected dataset, and P_IMERG,t is the corresponding 30-minute value from IMERG. This approach preserves both the corrected hourly total and the temporal distribution pattern observed by IMERG's more frequent observations.

#### Statistical Disaggregation
When IMERG data is unavailable, we employ a statistical approach based on regional precipitation intensity distributions:

```
P_30min,t = {
  0.6 × P_hour,  for first 30 minutes
  0.4 × P_hour,  for second 30 minutes
}
```

This 60/40 distribution reflects the typical front-loaded pattern of precipitation intensity within hourly intervals observed in the study region, with higher intensities typically occurring in the first half of each hour. For periods with zero precipitation, the disaggregated values remain zero to maintain physical consistency.

## Optimal Dataset Selection Methodology

For each station-event combination, a weighted composite score is calculated to determine the best satellite product. The methodology prioritizes datasets with strong temporal correlation patterns while maintaining reasonable error magnitudes. The composite score is calculated as:

```
Station Score = 2/(1+RMSE) + 2×Corr + 1/(1+|Bias|) + 1.5×KGE
```

This formulation incorporates:
- **Correlation with double weighting** (2×Corr) to emphasize temporal pattern matching
- **KGE with enhanced weighting** (1.5×KGE) as it encompasses correlation, bias, and variability
- **Inverse-weighted error terms** (2/(1+RMSE) and 1/(1+|Bias|)) to convert "lower is better" metrics to a consistent "higher is better" scoring framework

The dataset with the highest composite score for each station-event combination is selected as the optimal source for temporal disaggregation. This approach ensures that the final selection balances accurate temporal distribution patterns with acceptable error magnitudes.

## Precipitation Dataset Correction Methodology

After identifying the optimal satellite product for each station-event combination, we apply a correction procedure to address systematic biases while preserving temporal patterns. For each station and flood event, a scaling factor is derived using regression through the origin:

```
P_corrected = α × P_satellite
```

Where α represents the scaling factor that minimizes the difference between daily satellite estimates and ground observations. This approach maintains zero values and preserves relative intensity distribution within events. The scaling factors are constrained between 0.1 and 10.0 to prevent physically implausible adjustments.

For station-event combinations with fewer than five valid data pairs, we employ a simpler ratio-based method:

```
α_simple = Σ P_obs / Σ P_satellite
```

The effectiveness of scaling is quantified by comparing error metrics before and after correction:

```
Improvement % = (RMSE_original - RMSE_scaled) / RMSE_original × 100%
```

The correction process generates two products: 
1. A corrected daily dataset consistent with ground observations
2. A corrected high-resolution dataset where the same scaling factors are applied to sub-daily data, preserving temporal distribution while adjusting magnitude

This corrected high-resolution dataset serves as input for the subsequent temporal disaggregation procedure.

## Data Processing

### Unit Conversion and Aggregation

The tool handles different satellite data types automatically:

#### Rate Values (mm/hr)
- **GPM IMERG**: 30-minute precipitation rates
  - Conversion: `mm/hr × 0.5 hours = mm per 30-min period`
  - Daily total: Sum of all 30-minute periods
- **GSMaP**: Hourly precipitation rates  
  - Conversion: `mm/hr × 1 hour = mm per hour`
  - Daily total: Sum of all hourly values

#### Total/Accumulation Values (meters)
- **ERA5-Land**: Hourly precipitation accumulation
  - Conversion: `meters × 1000 = mm`
  - Daily total: Sum of all hourly accumulations

All datasets are automatically converted to consistent daily precipitation totals (mm) for comparison with station data.

## Output Files

The tool generates a ZIP package containing:

1. **analysis_summary.csv**: Statistical metrics for all station-dataset combinations
2. **corrected_[station]_[dataset].csv**: Bias-corrected high-resolution data for each combination
3. **README.txt**: Explanation of output files and methodology

## Sample Data

Sample data files are provided in the `temp/` directory:
- `sample_precipitation_data.csv`: Example daily precipitation data for 3 stations (STATION_001, STATION_002, STATION_003)
- `sample_station_metadata.csv`: Example station metadata for the same 3 stations

The sample data demonstrates the proper format:
- **Precipitation data**: Date column followed by station ID columns
- **Metadata**: Corresponding station information with matching IDs
- **Temporal coverage**: January 2020 sample period

## Dependencies

Required Python packages:
- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `earthengine-api`
- `scikit-learn` (optional, fallback implementations provided)

## Earth Engine Authentication

Ensure you have:
1. A Google Earth Engine account
2. Valid authentication setup
3. Proper project ID configuration

## Error Handling

The tool includes comprehensive error handling:
- **Data Validation**: Checks required columns and formats
- **Earth Engine Errors**: Automatic fallback to sample data for testing
- **Large Collections**: Chunked processing to prevent timeouts
- **Missing Dependencies**: Fallback implementations for statistical calculations

## Performance Considerations

- Large date ranges may require chunked processing
- Multiple stations increase processing time
- Earth Engine quotas may limit simultaneous requests
- Consider smaller time periods for initial testing

## Troubleshooting

### Common Issues

1. **Earth Engine Authentication**: Ensure proper GEE setup
2. **Large Data Requests**: Use smaller date ranges or fewer stations
3. **Missing Data**: Check data format and temporal coverage
4. **Slow Performance**: Earth Engine processing varies by server load

### Support

For technical support or questions:
- Check the main README.md for general setup instructions
- Verify Earth Engine authentication
- Test with provided sample data first
