#!/usr/bin/env python
"""
download_all.py — bootstrap all raw data sources for GeoSample AI.

Run once after cloning to populate data/raw/.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def download_sentinel2(args):
    from src.ingestion.sentinel2 import download_sentinel2
    download_sentinel2(
        output_dir=ROOT / "data/raw/sentinel2",
        config_path=ROOT / "configs/pipeline.yaml",
    )


def download_usgs(args):
    logger.info(
        "USGS Africa Mineral Geodatabase must be manually downloaded.\n"
        "  1. Visit: https://data.usgs.gov/datacatalog/data/USGS:607611a9d34e018b3201cbbf\n"
        "  2. Download Africa_GIS.gdb\n"
        f"  3. Place in: {ROOT / 'data/raw/usgs_geodatabase/'}\n"
        "\n"
        "USGS Copperbelt Database (SIR 2010-5090J):\n"
        "  1. Visit: https://pubs.usgs.gov/publication/sir20105090J\n"
        "  2. Download the appendix CSV/GDB files\n"
        f"  3. Place in: {ROOT / 'data/raw/copperbelt_db/'}"
    )


def download_ni43101(args):
    logger.info(
        "NI 43-101 reports are downloaded from SEDAR+ (public filing system).\n"
        "  1. Visit: https://www.sedarplus.ca\n"
        "  2. Search for companies: Ivanhoe Mines, First Quantum, Glencore (DRC/Zambia)\n"
        "  3. Filter: Filing Type = 'Technical Report'\n"
        f"  4. Save PDFs to: {ROOT / 'data/raw/ni43101_reports/'}\n"
        "\n"
        "Recommended starting reports:\n"
        "  - Ivanhoe Mines / Kamoa-Kakula IDP 2023 (DRC)\n"
        "  - First Quantum / Kansanshi Technical Report (Zambia)\n"
        "  - Glencore / Mopani Copper Mine (DRC)\n"
        "  - Barrick / Lumwana Technical Report (Zambia)"
    )


def main():
    parser = argparse.ArgumentParser(description="Download all GeoSample AI data sources")
    parser.add_argument(
        "--source",
        choices=["all", "sentinel2", "usgs", "ni43101"],
        default="all",
        help="Which data source to download (default: all)",
    )
    args = parser.parse_args()

    if args.source in ("all", "sentinel2"):
        logger.info("=== Sentinel-2 ===")
        download_sentinel2(args)

    if args.source in ("all", "usgs"):
        logger.info("=== USGS Geodatabases ===")
        download_usgs(args)

    if args.source in ("all", "ni43101"):
        logger.info("=== NI 43-101 Reports ===")
        download_ni43101(args)

    logger.info("Done. Check data/raw/ for downloaded files.")


if __name__ == "__main__":
    main()
