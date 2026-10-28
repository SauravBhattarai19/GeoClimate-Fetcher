"""
Multi-product ET ensemble and basin aggregation.

Global ET products disagree by large factors, so the module reports the spread
across products instead of a single "best" value. Membership is fixed over the
analysis period: only products that cover every month are ensemble members, so
products entering or leaving partway through cannot create artificial jumps.
"""

import datetime

import ee
import pandas as pd

from .grid import DEFAULT_RES_DEG, area_weighted_mean, to_grid
from .products import (
    ProductSpec,
    monthly_collection,
    native_projection,
    period_image,
    storage_collection,
)
from .temporal import add_months

ENSEMBLE_BANDS = ("median", "mean", "stdDev", "min", "max", "count", "cv")


def gridded_period_image(spec: ProductSpec, asset: str, months: list, months_per_period: int,
                         res: float = DEFAULT_RES_DEG) -> ee.Image:
    """A product's period total (mm) on the analysis grid."""
    return to_grid(
        period_image(spec, asset, months, months_per_period),
        native_projection(spec, asset),
        spec.native_res_deg,
        res,
    ).set("product", spec.key)


def ensemble_image(gridded: list) -> ee.Image:
    """
    Per-cell statistics across gridded product images (all single-band, same units).

    Bands: median, mean, stdDev, min, max, count (products with data) and cv
    (stdDev / mean in %, only where the mean exceeds 1 mm).
    """
    reducer = (
        ee.Reducer.median()
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )
    stack = ee.ImageCollection.fromImages([ee.Image(img).rename("v") for img in gridded])
    stats = stack.reduce(reducer).rename(list(ENSEMBLE_BANDS[:-1]))
    mean = stats.select("mean")
    cv = stats.select("stdDev").divide(mean).multiply(100).updateMask(mean.gt(1)).rename("cv")
    return stats.addBands(cv)


def basin_period_means(gridded: dict, geometry, res: float = DEFAULT_RES_DEG,
                       on_product=None) -> dict:
    """
    {product key: {'mean', 'coverage'}} for gridded period images.

    One request per product, so a product that times out or runs out of memory
    is reported as {'error': ...} without losing the others.
    on_product(done, total, key) reports progress.
    """
    out = {}
    for i, (key, img) in enumerate(gridded.items()):
        try:
            out[key] = area_weighted_mean(img, geometry, res).getInfo()
        except Exception as exc:
            out[key] = {"mean": None, "coverage": 0, "error": str(exc)[:300]}
        if on_product:
            on_product(i + 1, len(gridded), key)
    return out


def _chunk_size(spec: ProductSpec) -> int:
    """Months per request: fewer for fine-resolution products, which cost more to regrid."""
    return 12 if spec.native_res_deg < 0.02 else 36


def _basin_series(collection_fn, spec: ProductSpec, asset: str, months: list, geometry,
                  res: float, on_chunk=None) -> pd.DataFrame:
    """Area-weighted basin means of collection_fn(chunk of months), in chunks of months."""
    proj = native_projection(spec, asset)

    def summarize(img):
        gridded = to_grid(img, proj, spec.native_res_deg, res)
        return ee.Feature(None, area_weighted_mean(gridded, geometry, res)).set(
            "t", img.get("system:time_start")
        )

    rows = []
    step = _chunk_size(spec)
    for i in range(0, len(months), step):
        chunk = months[i:i + step]
        feats = collection_fn(chunk).map(summarize).getInfo()["features"]
        for feat in feats:
            props = feat["properties"]
            month = datetime.datetime.fromtimestamp(props["t"] / 1000, tz=datetime.timezone.utc).date()
            rows.append({"month": month, "mean": props.get("mean"),
                         "coverage": props.get("coverage", 0)})
        if on_chunk:
            on_chunk(min(i + step, len(months)), len(months))

    df = pd.DataFrame(rows).set_index("month").sort_index()
    df.index = pd.to_datetime(df.index)
    return df.astype(float)


def centred_change(storage: pd.Series) -> pd.Series:
    """
    Monthly storage change from a contiguous monthly storage series (same formula
    as products.monthly_image): [S(m+1) − S(m−1)] / 2, falling back to the
    forward or backward difference where one neighbour is missing.
    """
    nxt, prv = storage.shift(-1), storage.shift(1)
    return ((nxt - prv) / 2).fillna(nxt - storage).fillna(storage - prv)


def basin_monthly_series(spec: ProductSpec, asset: str, months: list, geometry,
                         res: float = DEFAULT_RES_DEG, on_chunk=None) -> pd.DataFrame:
    """
    Area-weighted basin mean of monthly totals (mm) on the analysis grid.

    Requests run in chunks of months so long records stay inside interactive
    time limits. on_chunk(done, total) reports progress. Returns a DataFrame
    indexed by month with columns 'mean' and 'coverage'; storage products add
    'storage' (basin-mean storage in mm) and give the storage change as 'mean'.
    `months` must be contiguous.
    """
    if spec.temporal != "storage":
        return _basin_series(lambda chunk: monthly_collection(spec, asset, chunk),
                             spec, asset, months, geometry, res, on_chunk)

    # Basin means are linear, so differencing basin-mean storage equals the basin
    # mean of the per-cell storage change, with one request instead of three.
    extended = [add_months(months[0], -1)] + list(months) + [add_months(months[-1], 1)]
    states = _basin_series(lambda chunk: storage_collection(spec, asset, chunk),
                           spec, asset, extended, geometry, res, on_chunk)
    out = pd.DataFrame({
        "mean": centred_change(states["mean"]),
        "coverage": states["coverage"],
        "storage": states["mean"],
    })
    return out.iloc[1:-1]


def ensemble_series(monthly: pd.DataFrame, members: list) -> pd.DataFrame:
    """
    Month-by-month ensemble statistics from a table of product columns (mm/month).

    Only `members` contribute; other columns are ignored.
    """
    cols = [m for m in members if m in monthly.columns]
    data = monthly[cols]
    return pd.DataFrame({
        "median": data.median(axis=1),
        "q25": data.quantile(0.25, axis=1),
        "q75": data.quantile(0.75, axis=1),
        "min": data.min(axis=1),
        "max": data.max(axis=1),
        "n": data.count(axis=1),
    }, index=monthly.index)


def annual_totals(monthly: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year totals (mm/yr), keeping only years with all 12 months present."""
    grouped = monthly.groupby(monthly.index.year)
    totals = grouped.sum(min_count=1)
    complete = grouped.count() == 12
    return totals.where(complete)
