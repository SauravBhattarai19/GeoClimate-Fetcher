"""
Band math engine for raster calculations.

Parses user expressions like (B4-B3)/(B4+B3) and evaluates them
using ee.Image.expression() on Earth Engine image collections.
"""

import re
import ee
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Common index presets with their expressions and required bands
INDEX_PRESETS = {
    "NDVI": {
        "expression": "(NIR - RED) / (NIR + RED)",
        "description": "Normalized Difference Vegetation Index",
        "bands": {"NIR": "Near-Infrared", "RED": "Red"},
        "typical_range": (-1, 1),
    },
    "NDWI": {
        "expression": "(GREEN - NIR) / (GREEN + NIR)",
        "description": "Normalized Difference Water Index",
        "bands": {"GREEN": "Green", "NIR": "Near-Infrared"},
        "typical_range": (-1, 1),
    },
    "NDBI": {
        "expression": "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "description": "Normalized Difference Built-up Index",
        "bands": {"SWIR1": "Short-Wave Infrared 1", "NIR": "Near-Infrared"},
        "typical_range": (-1, 1),
    },
    "EVI": {
        "expression": "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        "description": "Enhanced Vegetation Index",
        "bands": {"NIR": "Near-Infrared", "RED": "Red", "BLUE": "Blue"},
        "typical_range": (-1, 1),
    },
    "SAVI": {
        "expression": "1.5 * ((NIR - RED) / (NIR + RED + 0.5))",
        "description": "Soil Adjusted Vegetation Index",
        "bands": {"NIR": "Near-Infrared", "RED": "Red"},
        "typical_range": (-1, 1),
    },
    "NDSI": {
        "expression": "(GREEN - SWIR1) / (GREEN + SWIR1)",
        "description": "Normalized Difference Snow Index",
        "bands": {"GREEN": "Green", "SWIR1": "Short-Wave Infrared 1"},
        "typical_range": (-1, 1),
    },
    "BSI": {
        "expression": "((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))",
        "description": "Bare Soil Index",
        "bands": {"SWIR1": "Short-Wave Infrared 1", "RED": "Red", "NIR": "Near-Infrared", "BLUE": "Blue"},
        "typical_range": (-1, 1),
    },
    "MNDWI": {
        "expression": "(GREEN - SWIR1) / (GREEN + SWIR1)",
        "description": "Modified Normalized Difference Water Index",
        "bands": {"GREEN": "Green", "SWIR1": "Short-Wave Infrared 1"},
        "typical_range": (-1, 1),
    },
}


def extract_band_references(expression: str) -> List[str]:
    """
    Extract band variable names from an expression string.

    Recognizes patterns like: b1, b2, B1, B2, band1, NIR, RED, SWIR1, etc.
    Does NOT match pure numbers or decimal numbers like 2.5, 7.5.

    Args:
        expression: Band math expression string

    Returns:
        Sorted list of unique variable names found
    """
    # Find all word tokens that aren't pure numbers
    tokens = re.findall(r'\b([A-Za-z_]\w*)\b', expression)
    # Filter out common math functions/constants that aren't band names
    excluded = {'abs', 'sqrt', 'log', 'log10', 'exp', 'pow', 'min', 'max',
                'sin', 'cos', 'tan', 'pi', 'e', 'ceil', 'floor', 'round'}
    variables = sorted(set(t for t in tokens if t not in excluded))
    return variables


def validate_expression(expression: str, available_bands: List[str],
                        band_mapping: Dict[str, str]) -> Tuple[bool, str]:
    """
    Validate a band math expression.

    Args:
        expression: The expression string
        available_bands: List of actual band names in the dataset
        band_mapping: Mapping from expression variable -> actual band name

    Returns:
        (is_valid, error_message) tuple
    """
    if not expression or not expression.strip():
        return False, "Expression is empty."

    # Check balanced parentheses
    depth = 0
    for ch in expression:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            return False, "Unbalanced parentheses: extra closing ')'."
    if depth != 0:
        return False, "Unbalanced parentheses: missing closing ')'."

    # Extract referenced variables
    variables = extract_band_references(expression)

    if not variables:
        return False, "No band references found in expression."

    # Check all variables have mappings
    unmapped = [v for v in variables if v not in band_mapping]
    if unmapped:
        return False, f"Unmapped variables: {', '.join(unmapped)}. Assign each to a dataset band."

    # Check all mapped bands exist in dataset
    for var, band in band_mapping.items():
        if band not in available_bands:
            return False, f"Band '{band}' (mapped from '{var}') not found in dataset."

    # Try a simple syntax check by replacing variables with 1.0 and eval
    test_expr = expression
    for var in sorted(variables, key=len, reverse=True):
        test_expr = re.sub(r'\b' + re.escape(var) + r'\b', '1.0', test_expr)
    try:
        result = eval(test_expr)  # noqa: S307 - safe: only contains numbers and operators
        if not isinstance(result, (int, float)):
            return False, "Expression does not evaluate to a number."
    except ZeroDivisionError:
        # Division by zero is fine structurally (happens with test values)
        pass
    except Exception as e:
        return False, f"Invalid expression syntax: {e}"

    return True, ""


def apply_expression(image: ee.Image, expression: str,
                     band_mapping: Dict[str, str],
                     output_name: str = "result") -> ee.Image:
    """
    Apply a band math expression to an Earth Engine image.

    Uses ee.Image.expression() which supports standard math operators
    and common functions.

    Args:
        image: Input ee.Image
        expression: Math expression using variable names
        band_mapping: Variable name -> actual band name mapping
        output_name: Name for the output band

    Returns:
        ee.Image with a single band containing the result
    """
    # Build the band reference dict for ee.Image.expression()
    # Format: {'VAR': image.select('actual_band_name')}
    band_refs = {}
    for var_name, band_name in band_mapping.items():
        band_refs[var_name] = image.select(band_name)

    result = image.expression(expression, band_refs).rename(output_name)
    return result


def apply_expression_to_collection(
    collection: ee.ImageCollection,
    expression: str,
    band_mapping: Dict[str, str],
    output_name: str = "result"
) -> ee.ImageCollection:
    """
    Apply a band math expression to every image in a collection.

    Args:
        collection: Input ee.ImageCollection
        expression: Math expression
        band_mapping: Variable name -> actual band name mapping
        output_name: Name for the output band

    Returns:
        ee.ImageCollection where each image has a single result band
    """
    def compute(img):
        band_refs = {}
        for var_name, band_name in band_mapping.items():
            band_refs[var_name] = img.select(band_name)
        return img.expression(expression, band_refs).rename(output_name).copyProperties(img, ['system:time_start'])

    return collection.map(compute)


def aggregate_collection(collection: ee.ImageCollection,
                         method: str) -> ee.Image:
    """
    Aggregate an image collection into a single image.

    Args:
        collection: ee.ImageCollection to aggregate
        method: One of 'mean', 'median', 'min', 'max', 'sum', 'count'

    Returns:
        Single aggregated ee.Image
    """
    aggregators = {
        'mean': collection.mean,
        'median': collection.median,
        'min': collection.min,
        'max': collection.max,
        'sum': collection.sum,
        'count': collection.count,
    }

    if method not in aggregators:
        raise ValueError(f"Unknown aggregation method: {method}. "
                         f"Choose from: {', '.join(aggregators.keys())}")

    return aggregators[method]()
