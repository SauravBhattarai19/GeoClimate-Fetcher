"""
River-basin areas of interest from HydroSHEDS / HydroBASINS.

A water balance is computed per catchment, so the module lets users pick the
HydroBASINS unit containing a point and, optionally, everything upstream of it
(traced through the NEXT_DOWN topology).
"""

from collections import defaultdict

import ee

HYBAS_ASSET = "WWF/HydroSHEDS/v1/Basins/hybas_{level}"
BASIN_LEVELS = {
    3: "Level 3 — large river systems",
    4: "Level 4 — major basins",
    5: "Level 5",
    6: "Level 6 — sub-basins (recommended)",
    7: "Level 7",
    8: "Level 8 — small catchments",
}


def basin_collection(level: int) -> ee.FeatureCollection:
    return ee.FeatureCollection(HYBAS_ASSET.format(level=level))


def find_basin(lon: float, lat: float, level: int, upstream: bool = True) -> dict:
    """
    Locate the HydroBASINS unit containing (lon, lat).

    With upstream=True the result covers that unit plus every unit draining into it.
    Returns {'ids', 'hybas_id', 'level', 'area_km2', 'up_area_km2', 'n_units', 'upstream'},
    where up_area_km2 is HydroBASINS' own upstream area for the unit (a check on the trace);
    raises ValueError when the point is not inside any basin (e.g. ocean).
    """
    fc = basin_collection(level)
    hits = (
        fc.filterBounds(ee.Geometry.Point([lon, lat]))
        .reduceColumns(ee.Reducer.toList(4), ["HYBAS_ID", "MAIN_BAS", "SUB_AREA", "UP_AREA"])
        .getInfo()["list"]
    )
    if not hits:
        raise ValueError("No basin at this location (the point may be in the ocean).")
    hybas_id, main_bas, sub_area, up_area = int(hits[0][0]), hits[0][1], hits[0][2], hits[0][3]

    if not upstream:
        return {"ids": [hybas_id], "hybas_id": hybas_id, "level": level,
                "area_km2": sub_area, "up_area_km2": up_area, "n_units": 1, "upstream": False}

    # Trace upstream within the same main basin: invert NEXT_DOWN into child lists
    table = (
        fc.filter(ee.Filter.eq("MAIN_BAS", main_bas))
        .reduceColumns(ee.Reducer.toList(3), ["HYBAS_ID", "NEXT_DOWN", "SUB_AREA"])
        .getInfo()["list"]
    )
    children, area = defaultdict(list), {}
    for unit_id, next_down, sub_area in table:
        children[int(next_down)].append(int(unit_id))
        area[int(unit_id)] = sub_area

    ids, stack = [], [hybas_id]
    while stack:
        current = stack.pop()
        ids.append(current)
        stack.extend(children.get(current, []))

    return {"ids": ids, "hybas_id": hybas_id, "level": level,
            "area_km2": sum(area.get(i, 0) for i in ids), "up_area_km2": up_area,
            "n_units": len(ids), "upstream": True}


def basin_features(level: int, ids: list) -> ee.FeatureCollection:
    return basin_collection(level).filter(ee.Filter.inList("HYBAS_ID", ids))
