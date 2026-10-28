"""
Product registry for the Water Balance module.

Each ProductSpec knows how to turn its Earth Engine collection into calendar-month
totals in mm. Four temporal recipes cover every product:

- composite      8-day / dekadal images; each composite's daily rate is weighted by
                 the days it overlaps the month (gaps fall back to the valid days)
- mean_rate      daily or sub-daily rates (mm/day after `scale`); monthly mean × days
- monthly_total  one image per month already holding the monthly total
- storage        water storage states (mm); the month's storage change is the
                 centred difference ΔS(m) = [S(m+1) − S(m−1)] / 2, which matches
                 monthly-mean storage to a monthly flux. Where one neighbour is
                 missing (GRACE gaps) it falls back to a one-sided difference.

Unit facts (verified against the live catalog, 2026-10-27):
  MOD16A2GF ET     8-day SUM, ×0.1, values ≥32761 are fill (water/barren/urban)
  PML-V2.2a ET     mm/day ×0.01; open water/snow/ice only in Ew, so ET is filled from Ew
  WaPOR v3 AETI-D  mean mm/day over the dekad ×0.1, although labelled "mm"
  ERA5-Land        metres, negative = evaporation, monthly aggregate = monthly total
  GLDAS / FLDAS / SMAP L4   kg/m²/s (= mm/s)
  IMERG V07 monthly         mean rate in mm/hr
  CPC Unified      daily total in 0.1 mm (catalog unit column says "mm")
  CHIRPS / PERSIANN-CDR     mm/day, CHIRPS only 50°S–50°N, PERSIANN-CDR 60°S–60°N
  TerraClimate pr  monthly mm, no scale factor (aet/pet are ×0.1)
  Runoff           surface + subsurface: GLDAS-2.1 Qs_acc + Qsb_acc are kg/m² per 3 h;
                   GLDAS-2.2 / FLDAS / SMAP fluxes kg/m²/s; ERA5-Land metres;
                   TerraClimate ro mm with no scale factor
  GRACE mascon     cm of water-equivalent anomaly; image dates straddle calendar
                   months, so each image belongs to the month holding its midpoint
  GLDAS-2.2 TWS    mm (absolute storage; only differences are used)
"""

import datetime
from dataclasses import dataclass, field
from typing import Optional

import ee

from .temporal import (
    composite_end,
    first_full_month,
    last_full_month,
    months_to_millis,
    overlap_days,
)

SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class ProductSpec:
    key: str
    label: str
    kind: str                    # "et", "ref_et", "p", "q" (runoff) or "ds" (storage change)
    assets: tuple                # preferred asset first; later ones share the band schema
    band: str
    method: str                  # "Satellite" or "Model"
    algorithm: str
    native_res_deg: float
    native_res_label: str
    cadence: str                 # "8day", "dekadal", "monthly", "daily", "3hourly"
    temporal: str                # "composite", "mean_rate", "monthly_total", "storage"
    scale: float                 # raw value × scale → mm per composite / per day / per month
    color: str
    citation: str
    license: str
    url: str
    notes: str = ""
    period: Optional[str] = None           # composite period: "8day" or "dekad"
    composite_value: str = "rate"          # composites: "rate" (mm/day) or "sum" (mm/period)
    valid_max: Optional[float] = None      # raw values above this are fill codes
    fill_band: Optional[str] = None        # band that fills masked pixels of `band`
    add_bands: tuple = ()                  # bands added to `band` (e.g. surface + subsurface runoff)
    lat_bounds: Optional[tuple] = None     # (south, north) when the product is not global
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Actual evapotranspiration
# ---------------------------------------------------------------------------
ET_PRODUCTS = {
    "pml": ProductSpec(
        key="pml", label="PML-V2.2a", kind="et",
        assets=("projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a",),
        band="ET", fill_band="Ew",
        method="Satellite", algorithm="Penman-Monteith-Leuning, coupled with GPP",
        native_res_deg=0.0045, native_res_label="500 m",
        cadence="8day", temporal="composite", period="8day", composite_value="rate",
        scale=0.01, color="#1b9e77",
        citation=("Zhang, Y. et al. (2019). Coupled estimation of 500 m and 8-day resolution "
                  "global evapotranspiration and gross primary production in 2002–2017. "
                  "Remote Sensing of Environment, 222, 165–182."),
        license="CC-BY-4.0",
        url="https://developers.google.com/earth-engine/datasets/catalog/projects_pml_evapotranspiration_PML_OUTPUT_PML_V22a",
        notes="Transpiration + soil evaporation + interception; open water, snow and ice from Ew.",
    ),
    "mod16": ProductSpec(
        key="mod16", label="MOD16A2GF v6.1", kind="et",
        assets=("MODIS/061/MOD16A2GF",),
        band="ET", valid_max=32700,
        method="Satellite", algorithm="Penman-Monteith (MODIS)",
        native_res_deg=0.0045, native_res_label="500 m",
        cadence="8day", temporal="composite", period="8day", composite_value="sum",
        scale=0.1, color="#d95f02",
        citation=("Running, S., Mu, Q., Zhao, M., Moreno, A. (2021). MODIS/Terra Net "
                  "Evapotranspiration Gap-Filled 8-Day L4 Global 500m SIN Grid V061. "
                  "NASA LP DAAC. doi:10.5067/MODIS/MOD16A2GF.061"),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF",
        notes="No values over barren land, urban areas, water, snow and ice; "
              "tends to underestimate ET in drylands.",
    ),
    "ssebop": ProductSpec(
        key="ssebop", label="SSEBop VIIRS v6", kind="et",
        assets=("projects/usgs-ssebop/viirs_et_v6_monthly",),
        band="et",
        method="Satellite", algorithm="Operational Simplified Surface Energy Balance (thermal)",
        native_res_deg=0.00965, native_res_label="1 km",
        cadence="monthly", temporal="monthly_total",
        scale=1.0, color="#7570b3",
        citation=("Senay, G.B. et al. (2023). Improving the Operational Simplified Surface "
                  "Energy Balance Evapotranspiration Model Using the Forcing and Normalizing "
                  "Operation. Remote Sensing, 15(1), 260."),
        license="USGS public domain",
        url="https://gee-community-catalog.org/projects/usgs_viirs/",
        notes="Near-real-time (USGS FEWS NET); publisher-hosted asset.",
    ),
    "wapor": ProductSpec(
        key="wapor", label="WaPOR v3 L1", kind="et",
        assets=("FAO/WAPOR/3/L1_AETI_D",),
        band="L1-AETI-D",
        method="Satellite", algorithm="ETLook (FAO)",
        native_res_deg=0.003, native_res_label="300 m",
        cadence="dekadal", temporal="composite", period="dekad", composite_value="rate",
        scale=0.1, color="#e7298a",
        citation="FAO (2024). WaPOR Database Methodology, Version 3 Release. Rome, FAO.",
        license="FAO terms of use",
        url="https://developers.google.com/earth-engine/datasets/catalog/FAO_WAPOR_3_L1_AETI_D",
        notes="Actual ET and interception; global at Level 1 from 2018.",
    ),
    "era5land": ProductSpec(
        key="era5land", label="ERA5-Land", kind="et",
        assets=("ECMWF/ERA5_LAND/MONTHLY_AGGR",),
        band="total_evaporation_sum",
        method="Model", algorithm="ECMWF HTESSEL land reanalysis",
        native_res_deg=0.1, native_res_label="0.1°",
        cadence="monthly", temporal="monthly_total",
        scale=-1000.0, color="#66a61e",
        citation=("Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global "
                  "reanalysis dataset for land applications. Earth System Science Data, "
                  "13, 4349–4383."),
        license="Copernicus licence",
        url="https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR",
        notes="Includes inland-water evaporation. Sign flipped (ECMWF: negative = evaporation).",
    ),
    "gldas21": ProductSpec(
        key="gldas21", label="GLDAS-2.1 Noah", kind="et",
        assets=("NASA/GLDAS/V021/NOAH/G025/T3H",),
        band="Evap_tavg",
        method="Model", algorithm="Noah land surface model",
        native_res_deg=0.25, native_res_label="0.25°",
        cadence="3hourly", temporal="mean_rate",
        scale=SECONDS_PER_DAY, color="#e6ab02",
        citation=("Rodell, M. et al. (2004). The Global Land Data Assimilation System. "
                  "Bulletin of the American Meteorological Society, 85(3), 381–394."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H",
    ),
    "gldas22": ProductSpec(
        key="gldas22", label="GLDAS-2.2 CLSM-DA", kind="et",
        assets=("NASA/GLDAS/V022/CLSM/G025/DA1D",),
        band="Evap_tavg",
        method="Model", algorithm="Catchment land surface model with GRACE assimilation",
        native_res_deg=0.25, native_res_label="0.25°",
        cadence="daily", temporal="mean_rate",
        scale=SECONDS_PER_DAY, color="#a6761d",
        citation=("Li, B. et al. (2019). Global GRACE data assimilation for groundwater and "
                  "drought monitoring: Advances and challenges. Water Resources Research, "
                  "55, 7564–7586."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V022_CLSM_G025_DA1D",
    ),
    "fldas": ProductSpec(
        key="fldas", label="FLDAS Noah", kind="et",
        assets=("NASA/FLDAS/NOAH01/C/GL/M/V001",),
        band="Evap_tavg",
        method="Model", algorithm="Noah land surface model (MERRA-2 + CHIRPS forcing)",
        native_res_deg=0.1, native_res_label="0.1°",
        cadence="monthly", temporal="mean_rate",
        scale=SECONDS_PER_DAY, color="#666666",
        citation=("McNally, A. et al. (2017). A land data assimilation system for sub-Saharan "
                  "Africa food and water security applications. Scientific Data, 4, 170012."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001",
        notes="Set up for food-security monitoring; strongest in dryland regions.",
    ),
    "terraclimate": ProductSpec(
        key="terraclimate", label="TerraClimate", kind="et",
        assets=("IDAHO_EPSCOR/TERRACLIMATE",),
        band="aet",
        method="Model", algorithm="One-dimensional soil water balance (bucket) model",
        native_res_deg=1 / 24, native_res_label="~4 km",
        cadence="monthly", temporal="monthly_total",
        scale=0.1, color="#1f78b4",
        citation=("Abatzoglou, J.T., Dobrowski, S.Z., Parks, S.A., Hegewisch, K.C. (2018). "
                  "TerraClimate, a high-resolution global dataset of monthly climate and "
                  "climatic water balance from 1958–2015. Scientific Data, 5, 170191."),
        license="CC0-1.0",
        url="https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE",
        notes="Longest record (from 1958); best suited to long-term climatology.",
    ),
    "smap": ProductSpec(
        key="smap", label="SMAP L4", kind="et",
        assets=("NASA/SMAP/SPL4SMGP/008", "NASA/SMAP/SPL4SMGP/007"),
        band="land_evapotranspiration_flux",
        method="Model", algorithm="Catchment model with SMAP brightness-temperature assimilation",
        native_res_deg=0.095, native_res_label="9 km",
        cadence="3hourly", temporal="mean_rate",
        scale=SECONDS_PER_DAY, color="#b15928",
        citation=("Reichle, R.H. et al. (2019). Version 4 of the SMAP Level-4 Soil Moisture "
                  "Algorithm and Data Product. Journal of Advances in Modeling Earth Systems, "
                  "11, 3106–3130."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_008",
    ),
}

# ---------------------------------------------------------------------------
# Reference / potential evapotranspiration
# ---------------------------------------------------------------------------
REF_ET_PRODUCTS = {
    "agera5_et0": ProductSpec(
        key="agera5_et0", label="AgERA5 ET₀ (FAO-56)", kind="ref_et",
        assets=("projects/climate-engine-pro/assets/ce-ag-era5-v2/daily",),
        band="ReferenceET_PenmanMonteith_FAO56",
        method="Model", algorithm="FAO-56 Penman-Monteith, short-grass reference",
        native_res_deg=0.1, native_res_label="0.1°",
        cadence="daily", temporal="mean_rate",
        scale=1.0, color="#e31a1c",
        citation=("Boogaard, H. et al. (2020). Agrometeorological indicators from 1979 to "
                  "present derived from reanalysis. Copernicus Climate Change Service (C3S) "
                  "Climate Data Store. doi:10.24381/cds.6c68c9bb"),
        license="Copernicus licence",
        url="https://gee-community-catalog.org/projects/agera5_datasets/",
        notes="Hosted by Climate Engine; land only.",
    ),
    "terraclimate_pet": ProductSpec(
        key="terraclimate_pet", label="TerraClimate PET", kind="ref_et",
        assets=("IDAHO_EPSCOR/TERRACLIMATE",),
        band="pet",
        method="Model", algorithm="ASCE Penman-Monteith reference ET",
        native_res_deg=1 / 24, native_res_label="~4 km",
        cadence="monthly", temporal="monthly_total",
        scale=0.1, color="#fb9a99",
        citation=("Abatzoglou, J.T., Dobrowski, S.Z., Parks, S.A., Hegewisch, K.C. (2018). "
                  "TerraClimate. Scientific Data, 5, 170191."),
        license="CC0-1.0",
        url="https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE",
    ),
}

# ---------------------------------------------------------------------------
# Precipitation
# ---------------------------------------------------------------------------
P_PRODUCTS = {
    "chirps": ProductSpec(
        key="chirps", label="CHIRPS v2.0", kind="p",
        assets=("UCSB-CHG/CHIRPS/DAILY",),
        band="precipitation", lat_bounds=(-50, 50),
        method="Satellite + gauge", algorithm="Infrared cold-cloud duration blended with stations",
        native_res_deg=0.05, native_res_label="0.05°",
        cadence="daily", temporal="mean_rate",
        scale=1.0, color="#08519c",
        citation=("Funk, C. et al. (2015). The climate hazards infrared precipitation with "
                  "stations — a new environmental record for monitoring extremes. "
                  "Scientific Data, 2, 150066."),
        license="CC0-1.0",
        url="https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY",
        notes="50°S–50°N only. Strong in the tropics and drylands; tends to miss "
              "orographic and snowfall precipitation in high mountains.",
    ),
    "imerg": ProductSpec(
        key="imerg", label="IMERG V07 Final", kind="p",
        assets=("NASA/GPM_L3/IMERG_MONTHLY_V07",),
        band="precipitation",
        method="Satellite + gauge", algorithm="GPM multi-satellite merging, GPCC gauge-adjusted",
        native_res_deg=0.1, native_res_label="0.1°",
        cadence="monthly", temporal="mean_rate",
        scale=24.0, color="#6a51a3",
        citation=("Huffman, G.J. et al. (2023). GPM IMERG Final Precipitation L3 1 month "
                  "0.1 degree x 0.1 degree V07. NASA GES DISC. doi:10.5067/GPM/IMERG/3B-MONTH/07"),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_MONTHLY_V07",
        notes="Global; less reliable over snow and ice. Final run released ~4 months late.",
    ),
    "persiann": ProductSpec(
        key="persiann", label="PERSIANN-CDR", kind="p",
        assets=("NOAA/PERSIANN-CDR",),
        band="precipitation", lat_bounds=(-60, 60),
        method="Satellite + gauge", algorithm="Neural network on infrared, adjusted to GPCP monthly",
        native_res_deg=0.25, native_res_label="0.25°",
        cadence="daily", temporal="mean_rate",
        scale=1.0, color="#3182bd",
        citation=("Ashouri, H. et al. (2015). PERSIANN-CDR: Daily precipitation climate data "
                  "record from multisatellite observations for hydrological and climate studies. "
                  "Bulletin of the American Meteorological Society, 96(1), 69–83."),
        license="NOAA CDR (public domain)",
        url="https://developers.google.com/earth-engine/datasets/catalog/NOAA_PERSIANN-CDR",
        notes="60°S–60°N only; monthly totals follow GPCP.",
    ),
    "cpc": ProductSpec(
        key="cpc", label="CPC Unified (gauge)", kind="p",
        assets=("NOAA/CPC/Precipitation",),
        band="precipitation",
        method="Gauge", algorithm="Optimal interpolation of daily station reports",
        native_res_deg=0.5, native_res_label="0.5°",
        cadence="daily", temporal="mean_rate",
        scale=0.1, color="#252525",
        citation=("Chen, M. et al. (2008). Assessing objective techniques for gauge-based "
                  "analyses of global daily precipitation. Journal of Geophysical Research, "
                  "113, D04110."),
        license="NOAA (public domain)",
        url="https://developers.google.com/earth-engine/datasets/catalog/NOAA_CPC_Precipitation",
        notes="The only gauge-only product. Quality follows gauge density: poor over tropical "
              "Africa, deserts and polar regions (zero where no gauges report).",
    ),
    "era5land_p": ProductSpec(
        key="era5land_p", label="ERA5-Land", kind="p",
        assets=("ECMWF/ERA5_LAND/MONTHLY_AGGR",),
        band="total_precipitation_sum",
        method="Model", algorithm="ECMWF reanalysis (not gauge-corrected)",
        native_res_deg=0.1, native_res_label="0.1°",
        cadence="monthly", temporal="monthly_total",
        scale=1000.0, color="#66a61e",
        citation=("Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global "
                  "reanalysis dataset for land applications. Earth System Science Data, "
                  "13, 4349–4383."),
        license="Copernicus licence",
        url="https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR",
        notes="Rain plus snow from the ERA5 forecast model; tends to overestimate light rain "
              "and precipitation over mountains.",
    ),
    "gldas21_p": ProductSpec(
        key="gldas21_p", label="GLDAS-2.1 forcing", kind="p",
        assets=("NASA/GLDAS/V021/NOAH/G025/T3H",),
        band="Rainf_f_tavg",
        method="Satellite + gauge", algorithm="GPCP 1DD v1.3 disaggregated to 3-hourly with GDAS",
        native_res_deg=0.25, native_res_label="0.25°",
        cadence="3hourly", temporal="mean_rate",
        scale=SECONDS_PER_DAY, color="#e6ab02",
        citation=("Rodell, M. et al. (2004). The Global Land Data Assimilation System. "
                  "Bulletin of the American Meteorological Society, 85(3), 381–394."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H",
        notes="Total precipitation (rain + snow) forcing the GLDAS-2.1 Noah ET product.",
    ),
    "terraclimate_p": ProductSpec(
        key="terraclimate_p", label="TerraClimate", kind="p",
        assets=("IDAHO_EPSCOR/TERRACLIMATE",),
        band="pr",
        method="Gauge + reanalysis", algorithm="WorldClim climatology + CRU TS / JRA-55 anomalies",
        native_res_deg=1 / 24, native_res_label="~4 km",
        cadence="monthly", temporal="monthly_total",
        scale=1.0, color="#1f78b4",
        citation=("Abatzoglou, J.T., Dobrowski, S.Z., Parks, S.A., Hegewisch, K.C. (2018). "
                  "TerraClimate. Scientific Data, 5, 170191."),
        license="CC0-1.0",
        url="https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE",
        notes="Longest record (from 1958); interpolated, so trends follow the station network.",
    ),
}

# ---------------------------------------------------------------------------
# Runoff (surface + subsurface, generated per grid cell — not routed discharge)
# ---------------------------------------------------------------------------
Q_PRODUCTS = {
    "era5land_q": ProductSpec(
        key="era5land_q", label="ERA5-Land", kind="q",
        assets=("ECMWF/ERA5_LAND/MONTHLY_AGGR",),
        band="runoff_sum",
        method="Model", algorithm="ECMWF HTESSEL land reanalysis",
        native_res_deg=0.1, native_res_label="0.1°",
        cadence="monthly", temporal="monthly_total",
        scale=1000.0, color="#66a61e",
        citation=("Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global "
                  "reanalysis dataset for land applications. Earth System Science Data, "
                  "13, 4349–4383."),
        license="Copernicus licence",
        url="https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR",
        notes="Surface + subsurface runoff; ERA5 precipitation is not gauge-corrected.",
    ),
    "gldas21_q": ProductSpec(
        key="gldas21_q", label="GLDAS-2.1 Noah", kind="q",
        assets=("NASA/GLDAS/V021/NOAH/G025/T3H",),
        band="Qs_acc", add_bands=("Qsb_acc",),
        method="Model", algorithm="Noah land surface model",
        native_res_deg=0.25, native_res_label="0.25°",
        cadence="3hourly", temporal="mean_rate",
        scale=8.0, color="#e6ab02",
        citation=("Rodell, M. et al. (2004). The Global Land Data Assimilation System. "
                  "Bulletin of the American Meteorological Society, 85(3), 381–394."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H",
        notes="Storm surface runoff + baseflow, accumulated per 3 hours (×8 per day).",
    ),
    "gldas22_q": ProductSpec(
        key="gldas22_q", label="GLDAS-2.2 CLSM-DA", kind="q",
        assets=("NASA/GLDAS/V022/CLSM/G025/DA1D",),
        band="Qs_tavg", add_bands=("Qsb_tavg",),
        method="Model", algorithm="Catchment land surface model with GRACE assimilation",
        native_res_deg=0.25, native_res_label="0.25°",
        cadence="daily", temporal="mean_rate",
        scale=SECONDS_PER_DAY, color="#a6761d",
        citation=("Li, B. et al. (2019). Global GRACE data assimilation for groundwater and "
                  "drought monitoring: Advances and challenges. Water Resources Research, "
                  "55, 7564–7586."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V022_CLSM_G025_DA1D",
    ),
    "fldas_q": ProductSpec(
        key="fldas_q", label="FLDAS Noah", kind="q",
        assets=("NASA/FLDAS/NOAH01/C/GL/M/V001",),
        band="Qs_tavg", add_bands=("Qsb_tavg",),
        method="Model", algorithm="Noah land surface model (MERRA-2 + CHIRPS forcing)",
        native_res_deg=0.1, native_res_label="0.1°",
        cadence="monthly", temporal="mean_rate",
        scale=SECONDS_PER_DAY, color="#666666",
        citation=("McNally, A. et al. (2017). A land data assimilation system for sub-Saharan "
                  "Africa food and water security applications. Scientific Data, 4, 170012."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001",
        notes="Set up for food-security monitoring; strongest in dryland regions.",
    ),
    "terraclimate_q": ProductSpec(
        key="terraclimate_q", label="TerraClimate", kind="q",
        assets=("IDAHO_EPSCOR/TERRACLIMATE",),
        band="ro",
        method="Model", algorithm="One-dimensional soil water balance (bucket) model",
        native_res_deg=1 / 24, native_res_label="~4 km",
        cadence="monthly", temporal="monthly_total",
        scale=1.0, color="#1f78b4",
        citation=("Abatzoglou, J.T., Dobrowski, S.Z., Parks, S.A., Hegewisch, K.C. (2018). "
                  "TerraClimate. Scientific Data, 5, 170191."),
        license="CC0-1.0",
        url="https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE",
        notes="Water surplus of a bucket model; no groundwater or routing delay.",
    ),
    "smap_q": ProductSpec(
        key="smap_q", label="SMAP L4", kind="q",
        assets=("NASA/SMAP/SPL4SMGP/008", "NASA/SMAP/SPL4SMGP/007"),
        band="overland_runoff_flux", add_bands=("baseflow_flux",),
        method="Model", algorithm="Catchment model with SMAP brightness-temperature assimilation",
        native_res_deg=0.095, native_res_label="9 km",
        cadence="3hourly", temporal="mean_rate",
        scale=SECONDS_PER_DAY, color="#b15928",
        citation=("Reichle, R.H. et al. (2019). Version 4 of the SMAP Level-4 Soil Moisture "
                  "Algorithm and Data Product. Journal of Advances in Modeling Earth Systems, "
                  "11, 3106–3130."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_008",
    ),
}

# ---------------------------------------------------------------------------
# Terrestrial water storage change
# ---------------------------------------------------------------------------
DS_PRODUCTS = {
    "grace_jpl": ProductSpec(
        key="grace_jpl", label="GRACE/GRACE-FO JPL mascon", kind="ds",
        assets=("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI",),
        band="lwe_thickness",
        method="Satellite gravimetry", algorithm="JPL RL06.3 mascons, coastline-resolution filter",
        native_res_deg=0.5, native_res_label="0.5° (3° mascons)",
        cadence="monthly", temporal="storage",
        scale=10.0, color="#08306b",
        citation=("Watkins, M.M. et al. (2015). Improved methods for observing Earth's time "
                  "variable mass distribution with GRACE using spherical cap mascons. Journal "
                  "of Geophysical Research: Solid Earth, 120, 2648–2671."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_GRACE_MASS_GRIDS_V04_MASCON_CRI",
        notes="Effective resolution ~300 km: reliable for basins above ~200 000 km². "
              "No data Jul 2017 – May 2018 (gap between missions) and some single months.",
    ),
    "gldas22_tws": ProductSpec(
        key="gldas22_tws", label="GLDAS-2.2 CLSM-DA", kind="ds",
        assets=("NASA/GLDAS/V022/CLSM/G025/DA1D",),
        band="TWS_tavg",
        method="Model", algorithm="Catchment land surface model with GRACE assimilation",
        native_res_deg=0.25, native_res_label="0.25°",
        cadence="daily", temporal="storage",
        scale=1.0, color="#41ab5d",
        citation=("Li, B. et al. (2019). Global GRACE data assimilation for groundwater and "
                  "drought monitoring: Advances and challenges. Water Resources Research, "
                  "55, 7564–7586."),
        license="NASA open data",
        url="https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V022_CLSM_G025_DA1D",
        notes="Soil moisture, groundwater, snow and canopy storage. Assimilates GRACE, so not "
              "independent of it — but continues past GRACE and through its gaps.",
    ),
}

PRODUCTS_BY_KIND = {"et": ET_PRODUCTS, "ref_et": REF_ET_PRODUCTS, "p": P_PRODUCTS,
                    "q": Q_PRODUCTS, "ds": DS_PRODUCTS}
ALL_PRODUCTS = {**ET_PRODUCTS, **REF_ET_PRODUCTS, **P_PRODUCTS, **Q_PRODUCTS, **DS_PRODUCTS}


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def _utc(millis: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(millis / 1000, tz=datetime.timezone.utc).replace(tzinfo=None)


def fetch_availability(spec: ProductSpec) -> dict:
    """
    Query the live catalog for a product's date range, trying fallback assets in order.

    Returns {'asset', 'start', 'end', 'first_month', 'last_month'}; raises RuntimeError
    if no asset can be read.
    """
    errors = []
    for asset in spec.assets:
        try:
            rng = (
                ee.ImageCollection(asset)
                .reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
                .getInfo()
            )
            start, end = _utc(rng["min"]), _utc(rng["max"])
            return {
                "asset": asset,
                "start": start,
                "end": end,
                "first_month": first_full_month(start),
                "last_month": last_full_month(end, spec.cadence),
            }
        except Exception as exc:  # asset renamed, deprecated or access denied
            errors.append(f"{asset}: {exc}")
    raise RuntimeError(f"{spec.label} unavailable — " + "; ".join(errors))


def covers(availability: dict, first: datetime.date, last: datetime.date) -> bool:
    """True when the product has every month from first through last."""
    return availability["first_month"] <= first and last <= availability["last_month"]


def covers_latitudes(spec: ProductSpec, south: float, north: float) -> bool:
    """True when the product's latitude band contains the whole span south–north."""
    if spec.lat_bounds is None:
        return True
    return spec.lat_bounds[0] <= south and north <= spec.lat_bounds[1]


# ---------------------------------------------------------------------------
# Monthly harmonization (server-side)
# ---------------------------------------------------------------------------

def native_projection(spec: ProductSpec, asset: str) -> ee.Projection:
    return ee.ImageCollection(asset).first().select(spec.band).projection()


def _prepare(spec: ProductSpec, img: ee.Image) -> ee.Image:
    """Mask fill codes, fill from the companion band and apply the scale factor."""
    value = img.select(spec.band)
    for band in spec.add_bands:
        value = value.add(img.select(band))
    if spec.valid_max is not None:
        value = value.updateMask(value.lte(spec.valid_max))
    if spec.fill_band:
        value = value.unmask(img.select(spec.fill_band))
    return value.toFloat().multiply(spec.scale).rename("v").toFloat()


def _masked(bands: list) -> ee.Image:
    """Fully masked placeholder so reductions over an empty window still have bands."""
    return ee.Image.constant([0] * len(bands)).rename(bands).toFloat().updateMask(0)


def storage_image(spec: ProductSpec, asset: str, month_start) -> ee.Image:
    """
    Mean storage (mm) over the images whose time midpoint falls in the month, band 'S'.

    GRACE solutions span irregular windows (e.g. 25 Jan – 3 Mar is February), so
    filtering on start dates would put some months twice and leave others empty.
    """
    m0 = ee.Date(month_start)
    m1 = m0.advance(1, "month")

    def with_midpoint(img):
        start = ee.Number(img.get("system:time_start"))
        end = ee.Number(ee.Algorithms.If(img.get("system:time_end"), img.get("system:time_end"), start))
        return img.set("wb_mid", start.add(end).divide(2))

    states = (
        ee.ImageCollection(asset)
        .filterDate(m0.advance(-40, "day"), m1)
        .map(with_midpoint)
        .filter(ee.Filter.And(ee.Filter.gte("wb_mid", m0.millis()), ee.Filter.lt("wb_mid", m1.millis())))
        .map(lambda img: _prepare(spec, img))
    )
    return (
        states.merge(ee.ImageCollection([_masked(["v"])])).mean()
        .rename("S").toFloat().set("system:time_start", m0.millis())
    )


def storage_collection(spec: ProductSpec, asset: str, months: list) -> ee.ImageCollection:
    """Monthly storage states (mm) for a client-side list of first-of-month dates."""
    millis = ee.List(months_to_millis(months))
    return ee.ImageCollection.fromImages(
        millis.map(lambda t: storage_image(spec, asset, ee.Date(t)))
    )


def monthly_image(spec: ProductSpec, asset: str, month_start) -> ee.Image:
    """Total for one calendar month in mm, as a single float band named after spec.kind."""
    m0 = ee.Date(month_start)
    m1 = m0.advance(1, "month")
    ndays = m1.difference(m0, "day")
    col = ee.ImageCollection(asset)

    if spec.temporal == "storage":
        prev = storage_image(spec, asset, m0.advance(-1, "month"))
        cur = storage_image(spec, asset, m0)
        nxt = storage_image(spec, asset, m1)
        # unmask(image) takes the other image's value and mask where this one is masked
        total = (
            nxt.subtract(prev).divide(2)
            .unmask(nxt.subtract(cur))
            .unmask(cur.subtract(prev))
        )
    elif spec.temporal == "composite":
        def weighted(img):
            t0 = img.date()
            t1 = composite_end(t0, spec.period)
            rate = _prepare(spec, img)
            if spec.composite_value == "sum":
                rate = rate.divide(t1.difference(t0, "day"))
            days = overlap_days(t0, t1, m0, m1)
            weight = ee.Image.constant(days).toFloat().updateMask(rate.mask())
            return rate.multiply(days).rename("vw").addBands(weight.rename("w")).toFloat()

        sums = (
            col.filterDate(m0.advance(-10, "day"), m1)
            .map(weighted)
            .merge(ee.ImageCollection([_masked(["vw", "w"])]))
            .sum()
        )
        days_with_data = sums.select("w")
        total = (
            sums.select("vw").divide(days_with_data)
            .multiply(ndays)
            .updateMask(days_with_data.gt(0))
        )
    else:
        mean = (
            col.filterDate(m0, m1)
            .map(lambda img: _prepare(spec, img))
            .merge(ee.ImageCollection([_masked(["v"])]))
            .mean()
        )
        total = mean.multiply(ndays) if spec.temporal == "mean_rate" else mean

    return total.rename(spec.kind.upper()).toFloat().set("system:time_start", m0.millis())


def monthly_collection(spec: ProductSpec, asset: str, months: list) -> ee.ImageCollection:
    """Monthly totals (mm) for a client-side list of first-of-month dates."""
    millis = ee.List(months_to_millis(months))
    return ee.ImageCollection.fromImages(
        millis.map(lambda t: monthly_image(spec, asset, ee.Date(t)))
    )


def period_image(spec: ProductSpec, asset: str, months: list, months_per_period: int) -> ee.Image:
    """
    Mean monthly total × months_per_period, e.g. ×12 for an annual total.

    Averaging before scaling keeps an occasional missing month at a pixel from
    pulling the period total down. The result has no native projection, so it
    renders quickly at any zoom; pass native_projection() to to_grid() for analysis.
    """
    return (
        monthly_collection(spec, asset, months).mean()
        .multiply(months_per_period)
        .rename(spec.kind.upper())
    )
