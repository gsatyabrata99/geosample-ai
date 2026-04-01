"""
Sentinel-2 spectral band ratio analysis for copper alteration detection.

Computes band ratios sensitive to iron oxide and clay mineral zones
associated with copper porphyry and sediment-hosted deposits.

Key ratios:
  - Clay index: B11 / B12  (hydroxyl-bearing minerals)
  - Iron oxide: B11 / B8A  (ferric iron alteration)
  - Combined alteration score: normalised composite
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_clay_index(b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """
    Clay / hydroxyl mineral index: B11 / B12.
    High values indicate clay alteration zones.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b12 > 0, b11 / b12, np.nan)
    return ratio.astype(np.float32)


def compute_iron_oxide_index(b11: np.ndarray, b8a: np.ndarray) -> np.ndarray:
    """
    Iron oxide alteration index: B11 / B8A.
    Elevated values indicate ferric iron zones.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b8a > 0, b11 / b8a, np.nan)
    return ratio.astype(np.float32)


def normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1], ignoring NaNs."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return arr
    mn, mx = valid.min(), valid.max()
    if mx == mn:
        return np.zeros_like(arr)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def compute_alteration_score(
    b11: np.ndarray,
    b12: np.ndarray,
    b8a: np.ndarray,
    clay_weight: float = 0.5,
    iron_weight: float = 0.5,
) -> np.ndarray:
    """
    Composite alteration score combining clay and iron oxide indices.

    Returns:
        Array of values in [0, 1] where higher = stronger alteration signal.
    """
    clay = normalise(compute_clay_index(b11, b12))
    iron = normalise(compute_iron_oxide_index(b11, b8a))
    score = clay_weight * clay + iron_weight * iron
    return score.astype(np.float32)


def process_sentinel2_netcdf(
    nc_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Load a Sentinel-2 NetCDF (produced by the ingestion step),
    compute alteration scores, and save as GeoTIFF.

    Args:
        nc_path: Path to the sentinel2_alteration_bands.nc file.
        output_dir: Directory for output GeoTIFF.

    Returns:
        Path to the saved alteration score GeoTIFF.
    """
    try:
        import xarray as xr
        import rioxarray  # noqa: F401 — registers .rio accessor
    except ImportError:
        raise ImportError("Install xarray and rioxarray: pip install xarray rioxarray")

    nc_path = Path(nc_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Sentinel-2 bands from {nc_path} ...")
    ds = xr.open_dataset(nc_path)

    # Median composite across time to reduce cloud noise
    b11 = ds["B11"].median(dim="time").values
    b12 = ds["B12"].median(dim="time").values
    b8a = ds["B8A"].median(dim="time").values

    logger.info("Computing alteration score ...")
    score = compute_alteration_score(b11, b12, b8a)

    # Wrap back into xarray for easy GeoTIFF export
    score_da = xr.DataArray(
        score,
        dims=["y", "x"],
        coords={"y": ds["y"], "x": ds["x"]},
        name="alteration_score",
    )
    score_da = score_da.rio.write_crs(ds.rio.crs or "EPSG:32735")

    out_path = output_dir / "alteration_score.tif"
    score_da.rio.to_raster(out_path)
    logger.info(f"Saved alteration score raster to {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_sentinel2_netcdf(
        nc_path="data/raw/sentinel2/sentinel2_alteration_bands.nc",
        output_dir="data/processed/rasters",
    )
