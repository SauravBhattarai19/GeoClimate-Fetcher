"""
Data models for Earth Engine STAC catalog metadata.

This module defines dataclasses for representing dataset and band metadata
from the Google Earth Engine STAC API.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


@dataclass
class BandMetadata:
    """Represents detailed band information from STAC catalog."""

    name: str
    description: str = ""
    units: Optional[str] = None
    scale: Optional[float] = None
    offset: Optional[float] = None
    wavelength: Optional[str] = None
    center_wavelength: Optional[float] = None
    gsd: Optional[float] = None  # Ground sample distance in meters
    data_type: Optional[str] = None
    bitmask: Optional[Dict[str, Any]] = None
    classes: Optional[List[Dict[str, Any]]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    @classmethod
    def from_stac_dict(cls, band_dict: Dict[str, Any]) -> 'BandMetadata':
        """
        Create BandMetadata from STAC band dictionary.

        Args:
            band_dict: Dictionary from STAC summaries.eo:bands array

        Returns:
            BandMetadata instance
        """
        return cls(
            name=band_dict.get('name', ''),
            description=band_dict.get('description', ''),
            units=band_dict.get('gee:units'),
            scale=band_dict.get('gee:scale'),
            offset=band_dict.get('gee:offset'),
            wavelength=band_dict.get('gee:wavelength'),
            center_wavelength=band_dict.get('center_wavelength'),
            gsd=band_dict.get('gsd'),
            data_type=band_dict.get('data_type'),
            bitmask=band_dict.get('gee:bitmask'),
            classes=band_dict.get('gee:classes'),
            minimum=band_dict.get('minimum'),
            maximum=band_dict.get('maximum')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'units': self.units,
            'scale': self.scale,
            'offset': self.offset,
            'wavelength': self.wavelength,
            'center_wavelength': self.center_wavelength,
            'gsd': self.gsd,
            'data_type': self.data_type,
            'bitmask': self.bitmask,
            'classes': self.classes,
            'minimum': self.minimum,
            'maximum': self.maximum
        }


@dataclass
class DatasetMetadata:
    """Complete dataset metadata from STAC catalog."""

    id: str  # Earth Engine asset ID
    name: str  # Human-readable title
    description: str = ""
    provider: str = "Unknown"  # Primary provider name
    all_providers: List[Dict[str, str]] = field(default_factory=list)
    snippet_type: str = "ImageCollection"  # "Image" or "ImageCollection"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    temporal_resolution: str = "Unknown"
    pixel_size: Optional[float] = None  # meters
    bands: List[BandMetadata] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    license: str = "Unknown"
    stac_version: str = "1.0.0"
    category: str = "Other"
    spatial_extent: Optional[Dict[str, Any]] = None

    @classmethod
    def from_stac_collection(cls, collection: Dict[str, Any]) -> 'DatasetMetadata':
        """
        Create DatasetMetadata from STAC collection JSON.

        Args:
            collection: STAC collection dictionary

        Returns:
            DatasetMetadata instance
        """
        # Extract basic info
        dataset_id = collection.get('id', '')
        title = collection.get('title', dataset_id)
        description = collection.get('description', '')

        # Extract providers
        providers = collection.get('providers', [])
        primary_provider = "Unknown"
        if providers:
            # First provider is usually the primary one
            primary_provider = providers[0].get('name', 'Unknown')

        # Extract temporal extent
        start_date = None
        end_date = None
        extent = collection.get('extent', {})
        temporal = extent.get('temporal', {})
        interval = temporal.get('interval', [[None, None]])
        if interval and interval[0]:
            start_str = interval[0][0]
            end_str = interval[0][1]

            if start_str:
                try:
                    start_date = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                except:
                    pass

            if end_str:
                try:
                    end_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                except:
                    pass

        # Extract spatial extent
        spatial = extent.get('spatial', {})
        bbox = spatial.get('bbox', [[None, None, None, None]])
        spatial_extent = None
        if bbox and bbox[0] and all(x is not None for x in bbox[0]):
            spatial_extent = {
                'west': bbox[0][0],
                'south': bbox[0][1],
                'east': bbox[0][2],
                'north': bbox[0][3]
            }

        # Extract bands
        bands = []
        summaries = collection.get('summaries', {})
        band_list = summaries.get('eo:bands', [])
        for band_dict in band_list:
            bands.append(BandMetadata.from_stac_dict(band_dict))

        # Extract GSD (ground sample distance) as pixel size
        pixel_size = None
        gsd_list = summaries.get('gsd', [])
        if gsd_list:
            pixel_size = gsd_list[0] if isinstance(gsd_list, list) else gsd_list

        # Extract snippet type (gee:type)
        snippet_type = collection.get('gee:type', 'ImageCollection')

        # Extract keywords
        keywords = collection.get('keywords', [])

        # Extract license
        license_info = collection.get('license', 'Unknown')

        # Derive temporal resolution
        temporal_resolution = cls._derive_temporal_resolution(
            collection, keywords, start_date, end_date
        )

        # Derive category
        category = cls._derive_category(title, keywords, primary_provider)

        # STAC version
        stac_version = collection.get('stac_version', '1.0.0')

        return cls(
            id=dataset_id,
            name=title,
            description=description,
            provider=primary_provider,
            all_providers=providers,
            snippet_type=snippet_type,
            start_date=start_date,
            end_date=end_date,
            temporal_resolution=temporal_resolution,
            pixel_size=pixel_size,
            bands=bands,
            keywords=keywords,
            license=license_info,
            stac_version=stac_version,
            category=category,
            spatial_extent=spatial_extent
        )

    @staticmethod
    def _derive_temporal_resolution(
        collection: Dict[str, Any],
        keywords: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> str:
        """Derive temporal resolution from collection metadata."""
        # Check gee:revisit_interval
        gee_interval = collection.get('gee:revisit_interval', {})
        if gee_interval:
            interval_type = gee_interval.get('type', '').lower()
            interval_value = gee_interval.get('value', 1)

            if 'day' in interval_type:
                if interval_value == 1:
                    return 'Daily'
                else:
                    return f'{interval_value}-day'
            elif 'month' in interval_type:
                if interval_value == 1:
                    return 'Monthly'
                else:
                    return f'{interval_value}-month'
            elif 'year' in interval_type:
                return 'Annual'
            elif 'hour' in interval_type:
                return 'Hourly'

        # Check keywords for temporal hints
        keywords_lower = [k.lower() for k in keywords]

        if any(k in keywords_lower for k in ['daily', 'day']):
            return 'Daily'
        elif any(k in keywords_lower for k in ['monthly', 'month']):
            return 'Monthly'
        elif any(k in keywords_lower for k in ['annual', 'yearly', 'year']):
            return 'Annual'
        elif any(k in keywords_lower for k in ['hourly', 'hour']):
            return 'Hourly'
        elif any(k in keywords_lower for k in ['8-day', '8day']):
            return '8-day'
        elif any(k in keywords_lower for k in ['16-day', '16day']):
            return '16-day'

        # Check if dates are equal (static dataset)
        if start_date and end_date and start_date == end_date:
            return 'Static'

        # Default
        return 'Variable'

    @staticmethod
    def _derive_category(title: str, keywords: List[str], provider: str) -> str:
        """Derive category from title, keywords, and provider."""
        title_lower = title.lower()
        keywords_lower = ' '.join(keywords).lower()
        combined = f"{title_lower} {keywords_lower}"

        # Category mapping based on keywords and title
        category_keywords = {
            'Weather': ['weather', 'meteorology', 'atmospheric', 'wind', 'pressure'],
            'Climate': ['climate', 'temperature', 'precipitation', 'era5', 'reanalysis'],
            'Land Cover': ['land cover', 'landcover', 'land use', 'landuse', 'lulc'],
            'Vegetation': ['vegetation', 'ndvi', 'evi', 'modis', 'lai', 'biomass', 'forest'],
            'Ocean': ['ocean', 'sea surface', 'marine', 'bathymetry', 'chlorophyll'],
            'Atmosphere': ['aerosol', 'atmospheric', 'ozone', 'air quality', 'gas'],
            'Terrain': ['elevation', 'dem', 'topography', 'slope', 'terrain', 'srtm'],
            'Satellite Imagery': ['sentinel', 'landsat', 'aster', 'spot', 'imagery', 'optical'],
            'Radar': ['radar', 'sar', 'synthetic aperture'],
            'Hydrology': ['water', 'hydrology', 'river', 'flood', 'soil moisture'],
            'Geophysics': ['gravity', 'magnetic', 'geophysical'],
            'Population': ['population', 'demographic', 'settlement'],
            'Agriculture': ['crop', 'agriculture', 'farming', 'cropland']
        }

        for category, keywords_list in category_keywords.items():
            if any(keyword in combined for keyword in keywords_list):
                return category

        return 'Other'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility with CSV format."""
        return {
            'name': self.name,
            'ee_id': self.id,
            'snippet_type': self.snippet_type,
            'description': self.description,
            'category': self.category,
            'temporal_resolution': self.temporal_resolution,
            'provider': self.provider,
            'start_date': self.start_date.strftime('%Y-%m-%d') if self.start_date else '',
            'end_date': self.end_date.strftime('%Y-%m-%d') if self.end_date else '',
            'pixel_size': self.pixel_size if self.pixel_size else '',
            'band_names': ', '.join([band.name for band in self.bands]),
            'band_units': ', '.join([band.units or '' for band in self.bands]),
            'keywords': self.keywords,
            'license': self.license
        }

    def to_dataframe_row(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for pandas DataFrame row."""
        row = self.to_dict()
        # Add additional fields for DataFrame
        row['Dataset Name'] = self.name
        row['Earth Engine ID'] = self.id
        row['Snippet Type'] = self.snippet_type
        row['Description'] = self.description
        row['Category'] = self.category
        row['Temporal Resolution'] = self.temporal_resolution
        row['Provider'] = self.provider
        row['Start Date'] = self.start_date.strftime('%Y-%m-%d') if self.start_date else ''
        row['End Date'] = self.end_date.strftime('%Y-%m-%d') if self.end_date else ''
        row['Pixel Size (m)'] = self.pixel_size if self.pixel_size else ''
        row['Band Names'] = ', '.join([band.name for band in self.bands])
        row['Band Units'] = ', '.join([band.units or '' for band in self.bands])
        return row

    def get_band(self, band_name: str) -> Optional[BandMetadata]:
        """Get band metadata by name."""
        return next((band for band in self.bands if band.name == band_name), None)

    def get_band_names(self) -> List[str]:
        """Get list of band names."""
        return [band.name for band in self.bands]
