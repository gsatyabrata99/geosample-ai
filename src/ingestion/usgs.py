"""
USGS Africa Mineral Geodatabase ingestion.

Reads the Esri .gdb file, filters copper-related features,
and exports to GeoJSON for use in the scoring pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)

# USGS Africa GDB download URL (data.usgs.gov)
USGS_AFRICA_GDB_URL = (
    "https://doi.org/10.5066/P97EQWXP"
)

COPPER_KEYWORDS = [
    "copper", "cu", "cuprite", "chalcopyrite",
    "malachite", "azurite", "bornite", "chalcocite",
]

DEPOSIT_LAYER = "MineralOccurrenceSites"


def load_usgs_geodatabase(gdb_path: str | Path, output_dir: str | Path) -> gpd.GeoDataFrame:
    """
    Load the USGS Africa mineral geodatabase, filter for copper occurrences,
    and save to GeoJSON.

    Args:
        gdb_path: Path to the downloaded Africa_GIS.gdb file.
        output_dir: Where to save the filtered GeoJSON.

    Returns:
        GeoDataFrame of copper-related mineral occurrences.
    """
    gdb_path = Path(gdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gdb_path.exists():
        raise FileNotFoundError(
            f"GDB not found at {gdb_path}. "
            f"Download from: {USGS_AFRICA_GDB_URL}"
        )

    logger.info(f"Loading layer '{DEPOSIT_LAYER}' from {gdb_path} ...")
    gdf = gpd.read_file(gdb_path, layer=DEPOSIT_LAYER)
    logger.info(f"Loaded {len(gdf):,} total mineral occurrence records")

    # Filter for copper commodities (case-insensitive)
    copper_mask = gdf["COMMODITY"].str.lower().str.contains(
        "|".join(COPPER_KEYWORDS), na=False
    )
    copper_gdf = gdf[copper_mask].copy()
    logger.info(f"Filtered to {len(copper_gdf):,} copper-related occurrences")

    # Standardise CRS to WGS84
    copper_gdf = copper_gdf.to_crs("EPSG:4326")

    out_path = output_dir / "usgs_copper_deposits.geojson"
    copper_gdf.to_file(out_path, driver="GeoJSON")
    logger.info(f"Saved to {out_path}")

    return copper_gdf


def load_copperbelt_database(csv_path: str | Path, output_dir: str | Path) -> gpd.GeoDataFrame:
    """
    Load the USGS Central Africa Copperbelt deposit database (SIR 2010-5090J).
    Expects a CSV with LAT, LON, GRADE_PCT, TONNAGE_MT columns.

    Args:
        csv_path: Path to the downloaded Copperbelt CSV.
        output_dir: Where to save processed GeoJSON.

    Returns:
        GeoDataFrame with deposit-level attributes.
    """
    import pandas as pd
    from shapely.geometry import Point

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Copperbelt database from {csv_path} ...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} deposit records")

    # Build geometry from lat/lon columns (column names may vary — adjust as needed)
    lat_col = next(c for c in df.columns if "lat" in c.lower())
    lon_col = next(c for c in df.columns if "lon" in c.lower())

    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    out_path = output_dir / "copperbelt_deposits.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    logger.info(f"Saved {len(gdf):,} deposits to {out_path}")

    return gdf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_usgs_geodatabase(
        gdb_path="data/raw/usgs_geodatabase/Africa_GIS.gdb",
        output_dir="data/processed/vectors",
    )
