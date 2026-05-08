"""
SERVES v2.0 Calculator
Python/GEE translation of SERVES.java

Soil-moisture Estimation of Root zone through Vegetation index-based
Evapotranspiration fraction and Soil properties

Original algorithm: Dr. Nawa Raj Pradhan (ERDC/CHL, U.S. Army)
v2.0: Saurav Bhattarai, Dr. Rocky Talchabhadel, Dr. Nawa Raj Pradhan
      Jackson State University, Dept. of Civil & Environmental Engineering
"""

import ee
import datetime

# ==================================================================================
# SECTION 1: CONSTANTS & CONFIGURATION
# ==================================================================================

NDVI_COEFFICIENT = 1.33
NDVI_INTERCEPT = -0.049
LANDSAT_SCALE = 0.0000275
LANDSAT_OFFSET = -0.2
DEFAULT_SEARCH_WINDOW = 16
MAX_CLOUD_COVER = 100
DEFAULT_FIELD_CAPACITY = 0.35
DEFAULT_WILTING_POINT = 0.09
REGIONAL_SCALE = 500
WATER_THRESHOLD = 0
CHART_SCALE = 500
VERSION = "2.0.0"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

SEASON_LABELS = {
    "winter": "Winter (Dec–Feb)",
    "spring": "Spring (Mar–May)",
    "summer": "Summer (Jun–Aug)",
    "fall":   "Fall (Sep–Nov)",
}

SOIL_DEPTH_MAPPING = {
    "b0":   "val_0_5cm_mean",
    "b10":  "val_5_15cm_mean",
    "b30":  "val_15_30cm_mean",
    "b60":  "val_30_60cm_mean",
    "b100": "val_60_100cm_mean",
    "b200": "val_100_200cm_mean",
}

VIS_PARAMS = {
    "soil_moisture": {
        "min": 0.05,
        "max": 0.45,
        "palette": [
            "#8B0000", "#FF4500", "#FFA500", "#FFFF00",
            "#7CFC00", "#228B22", "#006400", "#00008B",
        ],
    },
    "soil_moisture_anomaly": {
        "min": -0.15,
        "max": 0.15,
        "palette": [
            "#8B0000", "#FF0000", "#FF6B6B", "#FFAAAA", "#FFFFFF",
            "#AAAAFF", "#6B6BFF", "#0000FF", "#00008B",
        ],
    },
    "ndvi": {
        "min": 0,
        "max": 0.8,
        "palette": [
            "#FFFFFF", "#CE7E45", "#DF923D", "#F1B555", "#FCD163",
            "#99B718", "#74A901", "#66A000", "#529400", "#3E8601",
            "#207401", "#056201", "#004C00", "#023B01", "#012E01",
            "#011D01", "#011301",
        ],
    },
    "et_fraction": {
        "min": 0,
        "max": 1,
        "palette": [
            "#FFFFE5", "#F7FCB9", "#D9F0A3", "#ADDD8E",
            "#78C679", "#41AB5D", "#238443", "#006837", "#004529",
        ],
    },
}

# Predefined region bounding boxes [west, south, east, north]
PREDEFINED_REGION_COORDS = {
    "globe":         [-180, -60,  180,  85],
    "europe":        [ -10,  35,   40,  71],
    "asia":          [  60, -10,  150,  55],
    "north_america": [-170,  15,  -50,  72],
    "south_america": [ -82, -56,  -34,  13],
    "africa":        [ -18, -35,   52,  37],
    "australia":     [ 113, -44,  154, -10],
}

PREDEFINED_REGION_LABELS = {
    "globe":         "🌍 Global",
    "europe":        "🇪🇺 Europe",
    "asia":          "🌏 Asia",
    "north_america": "🌎 North America",
    "south_america": "🌍 South America",
    "africa":        "🌍 Africa",
    "australia":     "🌏 Australia",
}


def get_month_name(month_num: int) -> str:
    return MONTH_NAMES[month_num - 1]


def get_predefined_region(name: str):
    """Return ee.Geometry.Rectangle for a named predefined region."""
    coords = PREDEFINED_REGION_COORDS[name]
    return ee.Geometry.Rectangle(coords)


# ==================================================================================
# SECTION 2: SOIL DATA FUNCTIONS
# ==================================================================================

def load_soil_data_soilgrids(study_area, depth_band: str = "b30") -> dict:
    """Load spatially variable FC & WP from ISRIC SoilGrids v2.0."""
    soil_band = SOIL_DEPTH_MAPPING.get(depth_band, "val_15_30cm_mean")

    field_capacity = (
        ee.Image("ISRIC/SoilGrids250m/v2_0/wv0033")
        .select(soil_band)
        .rename("field_capacity")
        .clip(study_area)
    )
    wilting_point = (
        ee.Image("ISRIC/SoilGrids250m/v2_0/wv1500")
        .select(soil_band)
        .rename("wilting_point")
        .clip(study_area)
    )
    return {
        "field_capacity": field_capacity,
        "wilting_point": wilting_point,
        "source": "SoilGrids250m v2.0",
        "depth": depth_band,
        "mode": "spatially_variable",
    }


def load_default_soil_parameters(study_area, fc_value=None, wp_value=None) -> dict:
    """Create uniform FC & WP constant images over the study area."""
    fc_value = fc_value if fc_value is not None else DEFAULT_FIELD_CAPACITY
    wp_value = wp_value if wp_value is not None else DEFAULT_WILTING_POINT

    field_capacity = ee.Image.constant(fc_value).rename("field_capacity").clip(study_area)
    wilting_point  = ee.Image.constant(wp_value).rename("wilting_point").clip(study_area)

    return {
        "field_capacity": field_capacity,
        "wilting_point": wilting_point,
        "source": "Uniform Default Values",
        "fc_value": fc_value,
        "wp_value": wp_value,
        "mode": "uniform",
    }


def load_soil_data(study_area, options: dict = None) -> dict:
    """Dispatch to SoilGrids or uniform defaults based on options."""
    options = options or {}
    mode = options.get("soil_parameter_mode", "soilgrids")
    if mode == "uniform":
        return load_default_soil_parameters(
            study_area,
            fc_value=options.get("field_capacity"),
            wp_value=options.get("wilting_point"),
        )
    return load_soil_data_soilgrids(study_area, options.get("soil_depth", "b30"))


# ==================================================================================
# SECTION 3: CLOUD MASKING & NDVI
# ==================================================================================

def mask_landsat_clouds(image):
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    return image.updateMask(mask)


def mask_s2_clouds(image):
    scl = image.select("SCL")
    clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return image.updateMask(clear)


def calculate_ndvi_landsat(image):
    nir  = image.select("SR_B5").multiply(LANDSAT_SCALE).add(LANDSAT_OFFSET)
    red  = image.select("SR_B4").multiply(LANDSAT_SCALE).add(LANDSAT_OFFSET)
    ndvi = nir.subtract(red).divide(nir.add(red)).clamp(-1, 1).rename("NDVI")
    return image.addBands(ndvi).copyProperties(image, image.propertyNames())


def calculate_ndvi_sentinel2(image):
    nir  = image.select("B8").divide(10000)
    red  = image.select("B4").divide(10000)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    return image.addBands(ndvi).copyProperties(image, image.propertyNames())


# ==================================================================================
# SECTION 4: DATE RANGE HELPERS
# ==================================================================================

def get_month_date_range(year, month: int) -> dict:
    start = ee.Date.fromYMD(year, month, 1)
    return {"start": start, "end": start.advance(1, "month")}


def get_season_date_range(year, season: str, hemisphere: str = "north") -> dict:
    if hemisphere == "north":
        season_map = {
            "winter": {"sm": 12, "syo": -1, "em": 3},
            "spring": {"sm":  3, "syo":  0, "em": 6},
            "summer": {"sm":  6, "syo":  0, "em": 9},
            "fall":   {"sm":  9, "syo":  0, "em": 12},
        }
    else:
        season_map = {
            "summer": {"sm": 12, "syo": -1, "em": 3},
            "fall":   {"sm":  3, "syo":  0, "em": 6},
            "winter": {"sm":  6, "syo":  0, "em": 9},
            "spring": {"sm":  9, "syo":  0, "em": 12},
        }
    s = season_map[season]
    start = ee.Date.fromYMD(year + s["syo"], s["sm"], 1)
    end   = ee.Date.fromYMD(year,            s["em"], 1)
    return {"start": start, "end": end}


# ==================================================================================
# SECTION 5: NDVI COLLECTION FUNCTIONS
# ==================================================================================

def get_ndvi_collection(study_area, start_date, end_date, satellite: str, options: dict = None):
    """Return clipped NDVI ImageCollection for the given satellite and date range."""
    options = options or {}
    start = ee.Date(start_date)
    end   = ee.Date(end_date)

    if satellite == "sentinel2":
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(study_area)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
            .map(mask_s2_clouds)
            .map(calculate_ndvi_sentinel2)
            .select("NDVI")
        )
    elif satellite == "modis":
        col = (
            ee.ImageCollection("MODIS/061/MOD13A2")
            .filterBounds(study_area)
            .filterDate(start, end)
            .map(
                lambda img: img.select("NDVI")
                .multiply(0.0001)
                .rename("NDVI")
                .copyProperties(img, img.propertyNames())
            )
        )
    else:  # landsat (default)
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(study_area).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD_COVER))
            .map(mask_landsat_clouds).map(calculate_ndvi_landsat).select("NDVI")
        )
        l9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(study_area).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD_COVER))
            .map(mask_landsat_clouds).map(calculate_ndvi_landsat).select("NDVI")
        )
        col = l8.merge(l9)

    return col.map(lambda img: img.clip(study_area))


def get_ndvi_single_date(study_area, target_date, satellite: str, search_window: int = None,
                          composite_method: str = "median") -> dict:
    """Get a single composite NDVI image near target_date."""
    search_window = search_window or DEFAULT_SEARCH_WINDOW
    target = ee.Date(target_date)
    start  = target.advance(-search_window, "day")
    end    = target.advance(search_window,  "day")

    if satellite == "sentinel2":
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(study_area).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
            .map(mask_s2_clouds).map(calculate_ndvi_sentinel2)
        )
    elif satellite == "modis":
        raw = (
            ee.ImageCollection("MODIS/061/MOD13A2")
            .filterBounds(study_area)
            .filterDate(target.advance(-16, "day"), target.advance(16, "day"))
        )
        with_dist = raw.map(
            lambda img: img.set(
                "date_diff",
                ee.Date(img.get("system:time_start")).difference(target, "day").abs()
            )
        )
        closest = ee.Image(with_dist.sort("date_diff").first())
        ndvi_img = closest.select("NDVI").multiply(0.0001).rename("NDVI").clip(study_area)
        return {"ndvi": ndvi_img, "image_count": raw.size(), "satellite": "MODIS Terra"}
    else:  # landsat
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(study_area).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD_COVER))
            .map(mask_landsat_clouds).map(calculate_ndvi_landsat)
        )
        l9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(study_area).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", MAX_CLOUD_COVER))
            .map(mask_landsat_clouds).map(calculate_ndvi_landsat)
        )
        coll = l8.merge(l9)

    image_count = coll.size()
    if composite_method == "closest":
        with_dist = coll.map(
            lambda img: img.set(
                "date_diff",
                ee.Date(img.get("system:time_start")).difference(target, "day").abs()
            )
        )
        ndvi_img = ee.Image(with_dist.sort("date_diff").first()).select("NDVI")
    elif composite_method == "mean":
        ndvi_img = coll.select("NDVI").mean()
    elif composite_method == "max":
        ndvi_img = coll.select("NDVI").max()
    else:
        ndvi_img = coll.select("NDVI").median()

    sat_labels = {"sentinel2": "Sentinel-2", "modis": "MODIS Terra"}
    return {
        "ndvi": ndvi_img.clip(study_area),
        "image_count": image_count,
        "satellite": sat_labels.get(satellite, "Landsat 8/9"),
    }


# ==================================================================================
# SECTION 6: SERVES CORE CALCULATION
# ==================================================================================

def calculate_serves(ndvi_image, soil_data: dict) -> dict:
    """Core SERVES formula: NDVI → ET fraction → soil moisture."""
    fc = soil_data["field_capacity"]
    wp = soil_data["wilting_point"]

    et_fraction = ndvi_image.multiply(NDVI_COEFFICIENT).add(NDVI_INTERCEPT).rename("et_fraction")
    et_frac_c   = et_fraction.clamp(0, 1).rename("et_fraction_constrained")

    paw          = fc.subtract(wp)
    soil_moisture = et_frac_c.multiply(paw).add(wp).rename("soil_moisture")
    soil_moisture = soil_moisture.where(soil_moisture.lt(wp), wp)
    soil_moisture = soil_moisture.where(soil_moisture.gt(fc), fc)

    deficit = fc.subtract(soil_moisture).rename("soil_moisture_deficit")

    return {
        "soil_moisture":          soil_moisture,
        "et_fraction":            et_fraction,
        "et_fraction_constrained": et_frac_c,
        "soil_moisture_deficit":  deficit,
        "ndvi":                   ndvi_image.rename("ndvi"),
        "field_capacity":         fc,
        "wilting_point":          wp,
        "plant_available_water":  paw.rename("plant_available_water"),
    }


# ==================================================================================
# SECTION 7: QUALITY MASKS
# ==================================================================================

def create_water_mask(study_area):
    water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").clip(study_area)
    return water.gt(WATER_THRESHOLD).rename("water_mask")


def create_vegetation_mask(study_area):
    wc = ee.Image("ESA/WorldCover/v200/2021").clip(study_area)
    return wc.eq(10).Or(wc.eq(20)).Or(wc.eq(30)).Or(wc.eq(40)).rename("vegetation_mask")


def apply_quality_masks(serves_result: dict, study_area, options: dict = None) -> dict:
    """Apply water/veg masks and special value assignments (v2.0 logic)."""
    options = options or {}
    mask_water              = options.get("mask_water", False)
    mask_non_veg            = options.get("mask_non_veg", False)
    assign_water_to_fc      = options.get("assign_water_to_fc", True)
    assign_neg_ndvi_to_fc   = options.get("assign_negative_ndvi_to_fc", True)

    sm   = serves_result["soil_moisture"]
    ndvi = serves_result["ndvi"]
    fc   = serves_result["field_capacity"]

    water_mask   = create_water_mask(study_area)
    veg_mask     = create_vegetation_mask(study_area)
    neg_ndvi_mask = ndvi.lt(0)

    if assign_water_to_fc:
        sm = sm.where(water_mask, fc)

    if assign_neg_ndvi_to_fc:
        sm = sm.where(neg_ndvi_mask, fc)

    combined_mask = ee.Image.constant(1)
    if mask_water and not assign_water_to_fc:
        combined_mask = combined_mask.multiply(water_mask.Not())
    if mask_non_veg:
        combined_mask = combined_mask.multiply(veg_mask)

    return {
        "soil_moisture":            sm.updateMask(combined_mask),
        "et_fraction_constrained":  serves_result["et_fraction_constrained"].updateMask(combined_mask),
        "soil_moisture_deficit":    serves_result["soil_moisture_deficit"].updateMask(combined_mask),
        "ndvi":                     ndvi.updateMask(combined_mask),
        "field_capacity":           fc,
        "wilting_point":            serves_result["wilting_point"],
        "plant_available_water":    serves_result["plant_available_water"],
        "quality_mask":             combined_mask.rename("quality_mask"),
    }


# ==================================================================================
# SECTION 8: MAIN ANALYSIS FUNCTIONS
# ==================================================================================

def run_serves(study_area, target_date: str, options: dict = None) -> dict:
    """Single-date SERVES analysis."""
    options = options or {}
    satellite       = options.get("satellite", "landsat")
    composite_method = options.get("composite_method", "median")
    search_window   = options.get("search_window", DEFAULT_SEARCH_WINDOW)

    ndvi_result = get_ndvi_single_date(
        study_area, target_date, satellite, search_window, composite_method
    )
    soil_data      = load_soil_data(study_area, options)
    serves_result  = calculate_serves(ndvi_result["ndvi"], soil_data)
    masked         = apply_quality_masks(serves_result, study_area, options)

    output_image = (
        masked["soil_moisture"]
        .addBands(masked["ndvi"])
        .addBands(masked["et_fraction_constrained"])
        .addBands(masked["soil_moisture_deficit"])
        .addBands(masked["field_capacity"])
        .addBands(masked["wilting_point"])
        .addBands(masked["plant_available_water"])
        .addBands(masked["quality_mask"])
    )

    return {
        "image":               output_image,
        "soil_moisture":       masked["soil_moisture"],
        "ndvi":                masked["ndvi"],
        "et_fraction":         masked["et_fraction_constrained"],
        "soil_moisture_deficit": masked["soil_moisture_deficit"],
        "field_capacity":      masked["field_capacity"],
        "wilting_point":       masked["wilting_point"],
        "quality_mask":        masked["quality_mask"],
        "metadata": {
            "satellite":        ndvi_result["satellite"],
            "image_count":      ndvi_result["image_count"],
            "composite_method": composite_method,
            "soil_source":      soil_data["source"],
            "target_date":      str(target_date),
            "mode":             "single",
        },
    }


def run_serves_for_period(study_area, start_date, end_date, options: dict = None) -> dict:
    """Period-averaged SERVES analysis (monthly / seasonal / annual / custom range)."""
    options   = options or {}
    satellite = options.get("satellite", "landsat")

    ndvi_col  = get_ndvi_collection(study_area, start_date, end_date, satellite, options)
    soil_data = load_soil_data(study_area, options)
    fc = soil_data["field_capacity"]
    wp = soil_data["wilting_point"]

    def apply_serves(image):
        ndvi    = image.select("NDVI")
        et_frac = ndvi.multiply(NDVI_COEFFICIENT).add(NDVI_INTERCEPT).clamp(0, 1)
        paw     = fc.subtract(wp)
        sm      = et_frac.multiply(paw).add(wp)
        sm      = sm.where(sm.lt(wp), wp)
        sm      = sm.where(sm.gt(fc), fc)
        return sm.rename("soil_moisture").copyProperties(image, ["system:time_start"])

    serves_col   = ndvi_col.map(apply_serves)
    sm_mean      = serves_col.mean().rename("soil_moisture")
    ndvi_mean    = ndvi_col.mean().rename("ndvi")

    mask_options = {
        "mask_water":               options.get("mask_water", False),
        "mask_non_veg":             options.get("mask_non_veg", False),
        "assign_water_to_fc":       options.get("assign_water_to_fc", True),
        "assign_negative_ndvi_to_fc": options.get("assign_negative_ndvi_to_fc", True),
    }
    serves_struct = {
        "soil_moisture":           sm_mean,
        "ndvi":                    ndvi_mean,
        "et_fraction_constrained": ee.Image.constant(0),
        "soil_moisture_deficit":   ee.Image.constant(0),
        "field_capacity":          fc,
        "wilting_point":           wp,
        "plant_available_water":   fc.subtract(wp),
    }
    masked = apply_quality_masks(serves_struct, study_area, mask_options)

    return {
        "soil_moisture": masked["soil_moisture"],
        "ndvi":          masked["ndvi"],
        "collection":    serves_col,
        "image_count":   ndvi_col.size(),
        "field_capacity": fc,
        "wilting_point":  wp,
        "metadata": {
            "satellite":   satellite,
            "soil_source": soil_data["source"],
            "mode":        "period",
        },
    }


def run_serves_time_series(study_area, start_date: str, end_date: str,
                            interval: str, options: dict = None) -> dict:
    """Time series SERVES at monthly / bi-weekly / weekly intervals."""
    options   = options or {}
    satellite = options.get("satellite", "landsat")

    soil_data = load_soil_data(study_area, options)
    fc = soil_data["field_capacity"]
    wp = soil_data["wilting_point"]

    ndvi_col = get_ndvi_collection(study_area, start_date, end_date, satellite, options)

    interval_days = 7 if interval == "weekly" else (16 if interval == "16day" else 30)

    fmt = "%Y-%m-%d"
    current = datetime.datetime.strptime(start_date, fmt)
    end_dt  = datetime.datetime.strptime(end_date,   fmt)

    images = []
    while current <= end_dt:
        date_str     = current.strftime(fmt)
        window_start = (current - datetime.timedelta(days=8)).strftime(fmt)
        window_end   = (current + datetime.timedelta(days=8)).strftime(fmt)

        ndvi_filtered = ndvi_col.filterDate(window_start, window_end)
        ndvi_median   = ndvi_filtered.median()

        et_frac = ndvi_median.multiply(NDVI_COEFFICIENT).add(NDVI_INTERCEPT).clamp(0, 1)
        paw     = fc.subtract(wp)
        sm      = et_frac.multiply(paw).add(wp)
        sm      = sm.where(sm.lt(wp), wp)
        sm      = sm.where(sm.gt(fc), fc)

        img = (
            sm.rename("soil_moisture")
            .set("system:time_start", ee.Date(date_str).millis())
            .set("date", date_str)
        )
        images.append(img)
        current += datetime.timedelta(days=interval_days)

    ts_collection = ee.ImageCollection(images) if images else ee.ImageCollection([])

    return {
        "image_collection": ts_collection,
        "start_date":       start_date,
        "end_date":         end_date,
        "interval":         interval,
        "num_steps":        len(images),
        "metadata": {
            "satellite":   satellite,
            "soil_source": soil_data["source"],
            "mode":        "time_series",
        },
    }


def run_serves_multi_year_monthly(study_area, start_year: int, end_year: int,
                                   month: int, options: dict = None) -> dict:
    """Multi-year monthly composite (one image per year, then mean)."""
    options = options or {}
    images = []
    for year in range(start_year, end_year + 1):
        dr     = get_month_date_range(year, month)
        result = run_serves_for_period(study_area, dr["start"], dr["end"], options)
        images.append(result["soil_moisture"].set("year", year))

    collection = ee.ImageCollection(images)
    return {
        "soil_moisture": collection.mean().rename("soil_moisture"),
        "collection":    collection,
        "image_count":   len(images),
        "start_year":    start_year,
        "end_year":      end_year,
        "month":         month,
    }


def run_serves_multi_year_seasonal(study_area, start_year: int, end_year: int,
                                    season: str, hemisphere: str = "north",
                                    options: dict = None) -> dict:
    """Multi-year seasonal composite."""
    options = options or {}
    images = []
    for year in range(start_year, end_year + 1):
        dr     = get_season_date_range(year, season, hemisphere)
        result = run_serves_for_period(study_area, dr["start"], dr["end"], options)
        images.append(result["soil_moisture"].set("year", year))

    collection = ee.ImageCollection(images)
    return {
        "soil_moisture": collection.mean().rename("soil_moisture"),
        "collection":    collection,
        "image_count":   len(images),
        "start_year":    start_year,
        "end_year":      end_year,
        "season":        season,
    }


def run_serves_multi_year_annual(study_area, start_year: int, end_year: int,
                                  options: dict = None) -> dict:
    """Multi-year annual composite."""
    options = options or {}
    images = []
    for year in range(start_year, end_year + 1):
        start  = ee.Date.fromYMD(year,     1, 1)
        end    = ee.Date.fromYMD(year + 1, 1, 1)
        result = run_serves_for_period(study_area, start, end, options)
        images.append(result["soil_moisture"].set("year", year))

    collection = ee.ImageCollection(images)
    return {
        "soil_moisture": collection.mean().rename("soil_moisture"),
        "collection":    collection,
        "image_count":   len(images),
        "start_year":    start_year,
        "end_year":      end_year,
    }


def run_serves_regional_climatology(region_name: str, start_year: int, end_year: int,
                                     month: int, options: dict = None) -> dict:
    """Regional climatology — forces MODIS satellite for large-scale analysis."""
    options = dict(options or {})
    options["satellite"] = "modis"  # Force MODIS for continental scale

    study_area = get_predefined_region(region_name)
    images = []
    for year in range(start_year, end_year + 1):
        dr     = get_month_date_range(year, month)
        result = run_serves_for_period(study_area, dr["start"], dr["end"], options)
        images.append(result["soil_moisture"].set("year", year))

    collection = ee.ImageCollection(images)
    return {
        "mean":       collection.mean().rename("mean_soil_moisture"),
        "min":        collection.min().rename("min_soil_moisture"),
        "max":        collection.max().rename("max_soil_moisture"),
        "median":     collection.median().rename("median_soil_moisture"),
        "std_dev":    collection.reduce(ee.Reducer.stdDev()).rename("stddev_soil_moisture"),
        "collection": collection,
        "study_area": study_area,
        "region":     region_name,
        "start_year": start_year,
        "end_year":   end_year,
        "month":      month,
        "year_count": len(images),
    }


# ==================================================================================
# SECTION 9: STATISTICS & TIME SERIES EXTRACTION
# ==================================================================================

def get_image_statistics(image, region, scale: int = None, band: str = "soil_moisture") -> dict:
    """Compute mean / min / max / stdDev of `band` over `region`. Returns plain dict."""
    scale = scale or CHART_SCALE
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.minMax(), "", True)
            .combine(ee.Reducer.stdDev(), "", True),
        geometry=region,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    )
    return stats.getInfo()


def extract_time_series_means(collection, region, scale: int = None,
                               band: str = "soil_moisture"):
    """
    Extract per-image mean of `band` from a time series ImageCollection.

    Returns (dates, means) where both are Python lists.
    Dates come from the 'date' property set on each image.
    None values (missing data) are kept so the caller can decide to skip them.
    """
    scale = scale or CHART_SCALE

    def add_mean(image):
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
        ).get(band)
        return image.set("mean_val", mean)

    with_means = collection.map(add_mean)
    dates = with_means.aggregate_array("date").getInfo()
    means = with_means.aggregate_array("mean_val").getInfo()
    return dates, means
