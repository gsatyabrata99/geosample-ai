# GeoSample AI

**AI-guided sample location optimizer for copper mineral exploration in Central and West Africa.**

GeoSample AI ingests unstructured geological data — PDF drill reports, field notes, satellite imagery, and georeferenced deposit databases — and outputs a ranked heatmap of optimal sampling sites, before anyone touches the ground.

---

## What it does

Junior mining companies spend $500K–$5M on exploration before knowing if a site is viable. Up to 70% of samples yield no actionable results. GeoSample AI cuts that waste by combining:

- **NLP on NI 43-101 reports** — custom NER extracts mineral names, ore grades, coordinates, and depth values from hundreds of PDFs
- **Sentinel-2 spectral analysis** — band ratio analysis on bands 11, 12, and 8A detects iron oxide and clay alteration zones associated with copper deposits
- **Gradient Boosting scoring** — trained on known USGS deposit locations, outputs a 0–1 site viability score
- **Ranked heatmap UI** — geologists see prioritized drill targets on an interactive map, with exportable reports

---

## Architecture

```
Data Sources        Processing           Intelligence         Platform
─────────────       ──────────────       ─────────────        ────────────
Sentinel-2      →   PDF/OCR          →   Custom NER       →   FastAPI
USGS GDB        →   Text cleaning    →   Topic modeling   →   React dashboard
NI 43-101 PDFs  →   Spectral bands   →   GBM scorer       →   Enterprise API
Copperbelt DB   →   Data Lake        →   Azure ML         →   Auth (Azure AD)
Field notes     →                    →                    →
```

---

## Project structure

```
geosample-ai/
├── data/
│   ├── raw/                    # Source data, never modified
│   │   ├── sentinel2/          # Sentinel-2 GeoTIFF tiles
│   │   ├── usgs_geodatabase/   # USGS Africa mineral .gdb files
│   │   ├── ni43101_reports/    # NI 43-101 PDF technical reports
│   │   ├── copperbelt_db/      # USGS Copperbelt deposit CSV/GDB
│   │   └── field_notes/        # Unstructured field note text
│   ├── processed/              # Cleaned, transformed outputs
│   │   ├── vectors/            # GeoJSON deposit locations
│   │   ├── rasters/            # Processed band ratio GeoTIFFs
│   │   └── text/               # Extracted and cleaned report text
│   └── features/               # ML-ready feature matrices
├── src/
│   ├── ingestion/              # Data downloaders and loaders
│   ├── processing/
│   │   ├── nlp/                # PDF extraction, cleaning, tokenization
│   │   └── spectral/           # Sentinel-2 band analysis
│   ├── intelligence/
│   │   ├── ner/                # Custom spaCy NER model
│   │   ├── topic_modeling/     # LDA across reports
│   │   └── scoring/            # GBM site viability model
│   └── platform/
│       ├── api/                # FastAPI backend
│       └── dashboard/          # React frontend
├── notebooks/                  # Exploratory analysis and demos
├── models/                     # Trained model artifacts
│   ├── ner/
│   ├── topic/
│   └── scoring/
├── configs/                    # YAML configuration files
├── scripts/                    # One-off utility scripts
├── tests/
│   ├── unit/
│   └── integration/
└── docs/                       # Architecture docs and API reference
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/gsatyabrata99/geosample-ai.git
cd geosample-ai

# Set up environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure credentials
cp configs/credentials.example.yaml configs/credentials.yaml
# Edit credentials.yaml with your Copernicus and Azure keys

# Download raw data
python scripts/download_all.py

# Run the pipeline
python scripts/run_pipeline.py --region drc --output data/features/
```

---

## Data sources

| Source | Format | License | Access |
|--------|--------|---------|--------|
| [Sentinel-2 L2A (Digital Earth Africa)](https://registry.opendata.aws/deafrica-sentinel-2/) | GeoTIFF | Open | Free, no auth |
| [USGS Africa Mineral Geodatabase](https://data.usgs.gov/datacatalog/data/USGS:607611a9d34e018b3201cbbf) | Esri GDB / Shapefile | Public domain | Free download |
| [USGS Copperbelt Database](https://pubs.usgs.gov/publication/sir20105090J) | CSV / GDB | Public domain | Free download |
| [NI 43-101 Technical Reports](https://www.sedar.com) | PDF | Public filing | Free (SEDAR) |
| Field notes | TXT / CSV | Proprietary | Client-provided |

---

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Lint
ruff check src/
```

---

## Roadmap

- [x] Repository structure and data source validation
- [ ] Data ingestion scripts (Sentinel-2, USGS, NI 43-101)
- [ ] NLP pipeline (PDF extraction, NER training, LDA)
- [ ] Spectral analysis (band ratios, alteration mapping)
- [ ] GBM scoring model
- [ ] FastAPI backend
- [ ] React dashboard with Mapbox heatmap
- [ ] Azure deployment (Azure ML, Azure AD, Azure Data Lake)
- [ ] Enterprise API and webhook integration

---

## Academic context

This project was initiated as part of CIS 8045 (Unstructured Data Management) at Georgia State University, Spring 2026, and is being developed into a production-grade enterprise product.

---

## License

MIT License — see [LICENSE](LICENSE)
