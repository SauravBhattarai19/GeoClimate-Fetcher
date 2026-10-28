"""
Precipitation minus evapotranspiration (P − ET), the first water-balance quantity.

Over a long period P − ET approximates basin runoff plus storage change
(P − ET = Q + ΔS). Both terms come from multi-product ensembles, so the
uncertainty of P − ET is taken from every pairing of a P member with an ET
member rather than from the two ensemble medians alone.
"""

import itertools

import ee
import pandas as pd

from .ensemble import ensemble_image, ensemble_series


def pair_differences(p_monthly: pd.DataFrame, et_monthly: pd.DataFrame,
                     p_members: list, et_members: list) -> pd.DataFrame:
    """Monthly P − ET (mm) for every (P member, ET member) pair, one column per pair."""
    pairs = {
        f"{p}|{et}": p_monthly[p] - et_monthly[et]
        for p, et in itertools.product(p_members, et_members)
        if p in p_monthly.columns and et in et_monthly.columns
    }
    return pd.DataFrame(pairs, index=p_monthly.index)


def balance_series(p_monthly: pd.DataFrame, et_monthly: pd.DataFrame,
                   p_members: list, et_members: list) -> pd.DataFrame:
    """
    Month-by-month P, ET and P − ET.

    Columns: p and et (ensemble medians) and median, q25, q75, min, max, n of
    P − ET across all member pairs.
    """
    diffs = pair_differences(p_monthly, et_monthly, p_members, et_members)
    out = ensemble_series(diffs, list(diffs.columns))
    out.insert(0, "et", ensemble_series(et_monthly, et_members)["median"])
    out.insert(0, "p", ensemble_series(p_monthly, p_members)["median"])
    return out


def balance_image(p_gridded: list, et_gridded: list) -> ee.Image:
    """Gridded P − ET (ensemble median P minus ensemble median ET), band 'p_minus_et'."""
    p = ensemble_image(p_gridded).select("median")
    et = ensemble_image(et_gridded).select("median")
    return p.subtract(et).rename("p_minus_et")


def discharge_to_monthly_mm(flow: pd.Series, area_km2: float, min_fraction: float = 0.8) -> pd.Series:
    """
    Monthly runoff depth (mm) from a river discharge record in m³/s.

    `flow` is indexed by timestamp at any step (sub-daily, daily or monthly).
    For daily or finer records, months with fewer than `min_fraction` of their
    days observed are dropped instead of being filled from a partial mean.
    depth = mean flow × seconds in month / basin area.
    """
    daily = flow.dropna().sort_index().resample("D").mean().dropna()
    monthly = daily.resample("MS").mean()
    days = monthly.index.days_in_month
    observed_days = daily.resample("MS").count()
    if observed_days.median() >= 25:    # a daily record rather than monthly values
        monthly = monthly.where(observed_days >= min_fraction * days)
    return (monthly * days * 86.4 / area_km2).rename("observed")   # m³/s·day/km² → mm
