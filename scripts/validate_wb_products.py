"""
Live validation of the Water Balance product harmonization.

Run against Earth Engine whenever products, assets or conversion code change:

    python scripts/validate_wb_products.py --project <gee-project-id>

Checks
  1. month arithmetic (offline)
  2. every product's live date range (and which asset answered)
  3. ERA5-Land sign/unit conversion matches the raw band
  4. composite products conserve the annual total: the 12 monthly totals of a
     year must equal the sum over that year's raw 8-day / dekadal composites
  5. grid alignment and area weighting
  6. July-2020 monthly ET of every product lies in a plausible range
  7. HydroBASINS lookup and upstream tracing
Exits non-zero if any check fails.
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ee  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}")


def check_months():
    from geoclimate_fetcher.water_balance.temporal import (
        add_months, first_full_month, last_full_month, month_range,
    )
    D, DT = datetime.date, datetime.datetime
    print("\n1. Month arithmetic")
    check("add_months across year end", add_months(D(2020, 11, 5), 3) == D(2021, 2, 1))
    check("month_range length", len(month_range(D(2019, 1, 1), D(2020, 12, 1))) == 24)
    check("first full month (mid-month start)", first_full_month(DT(2000, 3, 5)) == D(2000, 4, 1))
    check("first full month (1st)", first_full_month(DT(2003, 2, 1)) == D(2003, 2, 1))
    check("8-day: last composite of year closes December",
          last_full_month(DT(2025, 12, 27), "8day") == D(2025, 12, 1))
    check("dekadal: 2nd dekad leaves month incomplete",
          last_full_month(DT(2026, 10, 11), "dekadal") == D(2026, 9, 1))
    check("dekadal: 3rd dekad completes month",
          last_full_month(DT(2026, 9, 21), "dekadal") == D(2026, 9, 1))
    check("monthly", last_full_month(DT(2024, 12, 1), "monthly") == D(2024, 12, 1))
    check("daily", last_full_month(DT(2026, 8, 31), "daily") == D(2026, 8, 1))


def check_availability():
    from geoclimate_fetcher.water_balance import ALL_PRODUCTS, fetch_availability
    print("\n2. Live availability")
    out = {}
    for key, spec in ALL_PRODUCTS.items():
        try:
            av = fetch_availability(spec)
            out[key] = av
            check(spec.label, av["first_month"] <= av["last_month"],
                  f"{av['first_month']:%Y-%m} → {av['last_month']:%Y-%m}  ({av['asset']})")
        except Exception as exc:
            check(spec.label, False, str(exc)[:160])
    return out


def _point_value(img, lon, lat, scale):
    return ee.Image(img).reduceRegion(
        ee.Reducer.first(), ee.Geometry.Point([lon, lat]), scale
    ).values().get(0)


def check_era5_sign(avail):
    from geoclimate_fetcher.water_balance import ET_PRODUCTS, monthly_image
    print("\n3. ERA5-Land sign and units")
    spec = ET_PRODUCTS["era5land"]
    asset = avail["era5land"]["asset"]
    raw = ee.ImageCollection(asset).filterDate("2020-07-01", "2020-08-01").first().select(spec.band)
    harmonized = monthly_image(spec, asset, "2020-07-01")
    raw_v, mm_v = ee.List([
        _point_value(raw, -93.5, 42.0, 11132), _point_value(harmonized, -93.5, 42.0, 11132)
    ]).getInfo()
    check("Iowa July 2020: mm == −1000 × metres", abs(mm_v + 1000 * raw_v) < 1e-3,
          f"raw {raw_v:.4f} m → {mm_v:.1f} mm")


def check_composite_conservation(avail):
    from geoclimate_fetcher.water_balance import ET_PRODUCTS, monthly_collection
    from geoclimate_fetcher.water_balance.temporal import composite_end, month_range
    print("\n4. Composite → month conserves the annual total (2019, Iowa)")
    lon, lat, year = -93.5, 42.0, 2019
    months = month_range(datetime.date(year, 1, 1), datetime.date(year, 12, 1))
    for key in ("mod16", "pml", "wapor"):
        spec = ET_PRODUCTS[key]
        if key not in avail:
            check(spec.label, False, "unavailable")
            continue
        asset = avail[key]["asset"]
        if avail[key]["first_month"].year > year:
            year_used = avail[key]["first_month"].year
            months_used = month_range(datetime.date(year_used, 1, 1), datetime.date(year_used, 12, 1))
        else:
            year_used, months_used = year, months
        monthly_sum = monthly_collection(spec, asset, months_used).sum()

        def raw_total(img):
            t0 = img.date()
            t1 = composite_end(t0, spec.period)
            value = img.select(spec.band).toFloat().multiply(spec.scale)
            if spec.composite_value == "rate":
                value = value.multiply(t1.difference(t0, "day"))
            return value.rename("v")  # cropland test site: PML's Ew fill never applies

        raw = (
            ee.ImageCollection(asset)
            .filterDate(f"{year_used}-01-01", f"{year_used + 1}-01-01")
            .map(raw_total).sum()
        )
        scale = 500
        a, b = ee.List([_point_value(monthly_sum, lon, lat, scale),
                        _point_value(raw, lon, lat, scale)]).getInfo()
        ok = a is not None and b is not None and abs(a - b) <= 0.005 * max(abs(b), 1)
        check(f"{spec.label} {year_used}", ok, f"monthly sum {a:.1f} mm vs raw {b:.1f} mm"
              if a is not None and b is not None else f"{a} vs {b}")


def check_grid():
    from geoclimate_fetcher.water_balance import area_weighted_mean, crs_transform, to_grid
    print("\n5. Grid alignment and area weighting")
    img = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").first().select("temperature_2m")
    gridded = to_grid(img, img.projection(), 0.1, 0.1)
    transform = gridded.projection().getInfo()["transform"]
    check("ERA5-Land regridded onto (−180, 90) 0.1° grid",
          [round(v, 9) for v in transform] == crs_transform(0.1), str(transform))

    ones = ee.Image.constant(1).toFloat()
    box = ee.Geometry.Rectangle([10, 40, 20, 60], None, False)
    res = area_weighted_mean(ones, box, 0.1).getInfo()
    check("Constant field → mean 1, coverage 1",
          abs(res["mean"] - 1) < 1e-9 and abs(res["coverage"] - 1) < 1e-6, str(res))

    # Latitude gradient: area weighting must pull the mean toward the (larger) low-latitude cells
    lat_img = ee.Image.pixelLonLat().select("latitude")
    weighted = area_weighted_mean(lat_img, box, 0.1).getInfo()["mean"]
    check("Area weighting favours larger low-latitude cells", 49.0 < weighted < 50.0,
          f"mean latitude {weighted:.3f}° (unweighted would be 50°)")


def check_plausible_july(avail):
    from geoclimate_fetcher.water_balance import (
        ALL_PRODUCTS, area_weighted_mean, monthly_image, native_projection, to_grid,
    )
    print("\n6. July 2020 monthly totals, Iowa cropland 0.1° cell (plausible 60–250 mm)")
    cell = ee.Geometry.Rectangle([-93.6, 41.9, -93.5, 42.0], None, False)
    values = {}
    for key, spec in ALL_PRODUCTS.items():
        if key not in avail or not (avail[key]["first_month"] <= datetime.date(2020, 7, 1)
                                    <= avail[key]["last_month"]):
            continue
        asset = avail[key]["asset"]
        gridded = to_grid(monthly_image(spec, asset, "2020-07-01"),
                          native_projection(spec, asset), spec.native_res_deg, 0.1)
        values[key] = area_weighted_mean(gridded, cell, 0.1).get("mean")
    got = ee.Dictionary(values).getInfo()
    for key, v in got.items():
        check(ALL_PRODUCTS[key].label, v is not None and 60 <= v <= 250,
              f"{v:.1f} mm" if v is not None else "no data")


def check_basins():
    from geoclimate_fetcher.water_balance import find_basin
    print("\n7. HydroBASINS")
    single = find_basin(85.3, 27.7, 6, upstream=False)
    check("Level-6 unit at Kathmandu", abs(single["area_km2"] - 9894.8) < 1, f"{single['area_km2']} km²")
    up = find_basin(85.14, 25.62, 6, upstream=True)   # Ganges main stem at Patna
    check("Upstream trace matches HydroBASINS UP_AREA (Ganges at Patna)",
          up["n_units"] > 1 and abs(up["area_km2"] - up["up_area_km2"]) <= 0.01 * up["up_area_km2"],
          f"{up['n_units']} units, traced {up['area_km2']:,.0f} km² vs UP_AREA {up['up_area_km2']:,.0f} km²")
    try:
        find_basin(-30.0, 0.0, 6)
        check("Ocean point rejected", False)
    except ValueError:
        check("Ocean point rejected", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT"),
                        help="Earth Engine cloud project (default: $EE_PROJECT)")
    args = parser.parse_args()

    check_months()
    ee.Initialize(project=args.project)
    avail = check_availability()
    check_era5_sign(avail)
    check_composite_conservation(avail)
    check_grid()
    check_plausible_july(avail)
    check_basins()

    failed = RESULTS.count(False)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
