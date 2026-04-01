"""
Sentinel-2 data ingestion via Digital Earth Africa STAC API.

Pulls L2A tiles for the specified bounding box and date range,
selecting bands 11, 12, and 8A for copper alteration detection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/pipeline.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_stac_query(config: dict) -> dict:
    """Return parameters for a STAC catalog search."""
    s2 = config["sentinel2"]
    region = config["region"]
    return {
        "collections": [s2["collection"]],
        "bbox": region["bbox"],
        "datetime": f"{s2['date_range']['start']}/{s2['date_range']['end']}",
        "query": {"eo:cloud_cover": {"lt": s2["cloud_cover_max"]}},
    }


def download_sentinel2(output_dir: str | Path, config_path: str = "configs/pipeline.yaml") -> None:
    """
    Search the Digital Earth Africa STAC catalog and download Sentinel-2 tiles.

    Args:
        output_dir: Directory to save downloaded GeoTIFF files.
        config_path: Path to pipeline.yaml config.
    """
    try:
        from pystac_client import Client
        import odc.stac
    except ImportError:
        raise ImportError("Install pystac-client and odc-stac: pip install pystac-client odc-stac")

    config = load_config(config_path)
    s2_cfg = config["sentinel2"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Digital Earth Africa STAC endpoint...")
    catalog = Client.open(s2_cfg["stac_endpoint"])

    query_params = build_stac_query(config)
    logger.info(f"Searching STAC: bbox={config['region']['bbox']}, dates={s2_cfg['date_range']}")

    search = catalog.search(**query_params)
    items = list(search.items())
    logger.info(f"Found {len(items)} Sentinel-2 scenes")

    if not items:
        logger.warning("No scenes found. Check bbox, date range, or cloud cover threshold.")
        return

    bands = s2_cfg["bands"]
    resolution = s2_cfg["resolution"]

    logger.info(f"Loading bands {bands} at {resolution}m resolution...")
    ds = odc.stac.load(
        items,
        bands=bands,
        resolution=resolution,
        crs=f"EPSG:{config['region']['epsg']}",
        chunks={"time": 1, "x": 2048, "y": 2048},
    )

    out_path = output_dir / "sentinel2_alteration_bands.nc"
    logger.info(f"Saving to {out_path} ...")
    ds.to_netcdf(out_path)
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_sentinel2("data/raw/sentinel2")
