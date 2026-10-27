"""
Time handling for the Water Balance module.

The module's time step is the calendar month: every product is converted to a
monthly total in mm. Composite products (8-day, dekadal) are split across month
boundaries by the number of days each composite overlaps the month.
"""

import calendar
import datetime

import ee

MONTH_NAMES = list(calendar.month_name)[1:]

# Length of one composite period, used to find when a product's coverage ends
_PERIOD_DELTAS = {
    "8day": datetime.timedelta(days=8),
    "daily": datetime.timedelta(days=1),
    "3hourly": datetime.timedelta(hours=3),
}


# ---------------------------------------------------------------------------
# Client-side month arithmetic
# ---------------------------------------------------------------------------

def add_months(d: datetime.date, n: int) -> datetime.date:
    """First day of the month n months after d's month."""
    idx = d.year * 12 + (d.month - 1) + n
    return datetime.date(idx // 12, idx % 12 + 1, 1)


def month_range(first: datetime.date, last: datetime.date) -> list:
    """First-of-month dates from first's month through last's month, inclusive."""
    out, cur = [], datetime.date(first.year, first.month, 1)
    while cur <= last:
        out.append(cur)
        cur = add_months(cur, 1)
    return out


def first_full_month(min_dt: datetime.datetime) -> datetime.date:
    """First month entirely covered by data that starts at min_dt."""
    start = datetime.date(min_dt.year, min_dt.month, 1)
    return start if min_dt.day == 1 else add_months(start, 1)


def coverage_end(max_dt: datetime.datetime, cadence: str) -> datetime.datetime:
    """Exclusive end of the period covered by the last image (starting at max_dt)."""
    if cadence == "monthly":
        nxt = add_months(max_dt.date(), 1)
        return datetime.datetime(nxt.year, nxt.month, 1)
    if cadence == "dekadal":
        if max_dt.day >= 21:
            nxt = add_months(max_dt.date(), 1)
            return datetime.datetime(nxt.year, nxt.month, 1)
        return max_dt + datetime.timedelta(days=10)
    return max_dt + _PERIOD_DELTAS[cadence]


def last_full_month(max_dt: datetime.datetime, cadence: str) -> datetime.date:
    """Last month entirely covered by data whose last image starts at max_dt."""
    end = coverage_end(max_dt, cadence)
    return add_months(datetime.date(end.year, end.month, 1), -1)


def months_to_millis(months: list) -> list:
    return [
        int(datetime.datetime(m.year, m.month, 1, tzinfo=datetime.timezone.utc).timestamp() * 1000)
        for m in months
    ]


# ---------------------------------------------------------------------------
# Server-side helpers
# ---------------------------------------------------------------------------

def composite_end(start: ee.Date, period: str) -> ee.Date:
    """
    Exclusive end of a composite starting at `start`.

    8-day composites restart on 1 January, so the last one each year is 5–6 days.
    Dekads start on days 1, 11 and 21; the third runs to the end of the month.
    """
    if period == "8day":
        nxt = start.advance(8, "day")
        year_end = ee.Date.fromYMD(start.get("year").add(1), 1, 1)
        return ee.Date(ee.Algorithms.If(nxt.millis().gt(year_end.millis()), year_end, nxt))
    if period == "dekad":
        month_end = ee.Date.fromYMD(start.get("year"), start.get("month"), 1).advance(1, "month")
        return ee.Date(ee.Algorithms.If(start.get("day").lt(21), start.advance(10, "day"), month_end))
    raise ValueError(f"Unknown composite period: {period}")


def overlap_days(t0: ee.Date, t1: ee.Date, m0: ee.Date, m1: ee.Date) -> ee.Number:
    """Days of [t0, t1) that fall inside [m0, m1)."""
    s = ee.Date(ee.Algorithms.If(t0.millis().gt(m0.millis()), t0, m0))
    e = ee.Date(ee.Algorithms.If(t1.millis().lt(m1.millis()), t1, m1))
    return e.difference(s, "day").max(0)
