#!/usr/bin/env python
"""
run_pipeline.py — execute the full GeoSample AI pipeline end-to-end.

Steps:
  1. Extract text from NI 43-101 PDFs
  2. Compute Sentinel-2 alteration scores
  3. Load and filter USGS deposit vectors
  4. (stub) Run NER, LDA, GBM scoring
  5. Output ranked site heatmap GeoJSON
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_pipeline(region: str, config_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting GeoSample AI pipeline — region: {region}")

    # Step 1: PDF text extraction
    logger.info("[1/5] Extracting NI 43-101 report text ...")
    from src.ingestion.ni43101 import process_report_directory
    process_report_directory(
        pdf_dir=ROOT / "data/raw/ni43101_reports",
        output_dir=ROOT / "data/processed/text",
    )

    # Step 2: Sentinel-2 spectral analysis
    logger.info("[2/5] Computing Sentinel-2 alteration scores ...")
    nc_path = ROOT / "data/raw/sentinel2/sentinel2_alteration_bands.nc"
    if nc_path.exists():
        from src.processing.spectral.band_analysis import process_sentinel2_netcdf
        process_sentinel2_netcdf(
            nc_path=nc_path,
            output_dir=ROOT / "data/processed/rasters",
        )
    else:
        logger.warning(f"Sentinel-2 data not found at {nc_path} — run download_all.py first")

    # Step 3: USGS vector loading
    logger.info("[3/5] Loading USGS deposit vectors ...")
    gdb_path = ROOT / "data/raw/usgs_geodatabase/Africa_GIS.gdb"
    if gdb_path.exists():
        from src.ingestion.usgs import load_usgs_geodatabase
        load_usgs_geodatabase(
            gdb_path=gdb_path,
            output_dir=ROOT / "data/processed/vectors",
        )
    else:
        logger.warning(f"USGS GDB not found at {gdb_path} — see scripts/download_all.py")

    # Steps 4-5: NER, LDA, GBM scoring — stubs until models are trained
    logger.info("[4/5] NER + topic modeling — not yet implemented")
    logger.info("[5/5] GBM scoring + heatmap output — not yet implemented")

    logger.info(f"Pipeline complete. Outputs in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run the GeoSample AI pipeline")
    parser.add_argument("--region", default="drc", help="Region identifier (default: drc)")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Config file path")
    parser.add_argument("--output", default="data/features", help="Output directory")
    args = parser.parse_args()

    run_pipeline(region=args.region, config_path=args.config, output_dir=args.output)


if __name__ == "__main__":
    main()
