from setuptools import setup, find_packages

setup(
    name="geoclimate_fetcher",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "earthengine-api",
        "geemap",
        "pandas",
        "xarray",
        "streamlit",
        "folium",
        "streamlit-folium",
    ],
    author="ASCE EWRI",
    author_email="your.email@example.com",
    description="A tool for fetching climate data from Google Earth Engine",
    keywords="climate, earth engine, geospatial",
    python_requires=">=3.8",
) 