"""
Common analysis grid for the Water Balance module.

Every product is brought onto one fixed WGS84 grid whose cell edges are anchored
at (-180°, 90°), so cells from different products line up exactly:

- products finer than the grid are averaged into it (area-weighted by fractional
  overlap); a cell is kept only when enough of it holds valid fine pixels
- products coarser than (or equal to) the grid's cells are sampled
  nearest-neighbour, which invents no detail and keeps basin totals intact

Basin means are weighted by true cell area, since a 0.1° cell shrinks with latitude.
"""

import ee

GRID_CRS = "EPSG:4326"
DEFAULT_RES_DEG = 0.1
GRID_OPTIONS = (0.05, 0.1, 0.25, 0.5)
MIN_VALID_FRACTION = 0.5     # share of a grid cell that must hold valid fine pixels
MAX_INPUT_PIXELS = 65535     # reduceResolution limit (0.5° from 300 m ≈ 34 000 pixels)
METERS_PER_DEGREE = 111319.49


def crs_transform(res: float = DEFAULT_RES_DEG) -> list:
    """Affine transform of the grid: cell edges on multiples of res from (-180, 90)."""
    return [res, 0, -180, 0, -res, 90]


def grid_projection(res: float = DEFAULT_RES_DEG) -> ee.Projection:
    return ee.Projection(GRID_CRS, crs_transform(res))


def res_to_meters(res: float) -> float:
    """Nominal export scale (m) for a grid resolution in degrees."""
    return res * METERS_PER_DEGREE


def to_grid(image, native_proj, native_res_deg: float, res: float = DEFAULT_RES_DEG,
            min_valid: float = MIN_VALID_FRACTION) -> ee.Image:
    """
    Resample a single-band image onto the analysis grid.

    native_proj is required because composites (collection sums/means) fall back
    to a 1° default projection, which would silently break reduceResolution.
    """
    image = ee.Image(image).setDefaultProjection(native_proj)
    target = grid_projection(res)

    # Equal resolution still averages: ERA5-Land/AgERA5 cells are centred on 0.1°
    # multiples, half a cell off this grid, and nearest-neighbour would shift them.
    if native_res_deg <= res:
        valid = image.mask().unmask(0).rename("valid")
        averaged = (
            image.rename("v").addBands(valid)
            .reduceResolution(ee.Reducer.mean(), maxPixels=MAX_INPUT_PIXELS)
            .reproject(target)
        )
        return (
            averaged.select("v")
            .updateMask(averaged.select("valid").gte(min_valid))
            .rename(image.bandNames())
        )
    return image.reproject(target)


def area_weighted_mean(image, geometry, res: float = DEFAULT_RES_DEG) -> ee.Dictionary:
    """
    Area-weighted mean of a gridded single-band image over a geometry.

    Returns {'mean': value or None, 'coverage': fraction of the area with data}.
    Boundary cells count by the fraction that lies inside the geometry.
    """
    image = ee.Image(image)
    area = ee.Image.pixelArea()
    stack = (
        image.multiply(area).rename("w")
        .addBands(area.updateMask(image.mask()).rename("a"))
        .addBands(area.rename("t"))
    )
    sums = stack.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        crs=GRID_CRS,
        crsTransform=crs_transform(res),
        maxPixels=1e13,
        # 16 keeps MOD16's sinusoidal→lat/lon regridding inside memory on large basins
        # (Ganges, 788 000 km² fails at 4) and was faster for PML and WaPOR too
        tileScale=16,
    )
    covered = ee.Number(sums.get("a"))
    has_data = covered.gt(0)
    return ee.Dictionary({
        "mean": ee.Algorithms.If(has_data, ee.Number(sums.get("w")).divide(covered), None),
        "coverage": ee.Algorithms.If(
            ee.Number(sums.get("t")).gt(0), covered.divide(sums.get("t")), 0
        ),
    })
