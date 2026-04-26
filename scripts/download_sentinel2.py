"""
Download Sentinel-2 L2A tiles for the DRC Copperbelt region.
Bands: B11, B12, B8A (SWIR — copper alteration detection)
Area: Kamoa-Kakula licence area, DRC
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.sentinel2 import download_sentinel2

if __name__ == "__main__":
    download_sentinel2(
        output_dir="data/raw/sentinel2",
        config_path="configs/pipeline.yaml",
    )
    print("Done. Check data/raw/sentinel2/")
