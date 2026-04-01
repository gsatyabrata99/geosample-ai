# Architecture

## Overview

GeoSample AI is a four-layer system: data ingestion → processing → intelligence → platform.
Each layer is independently testable and deployable.

## Layer 1 — Data ingestion (`src/ingestion/`)

| Module | Source | Output |
|--------|--------|--------|
| `sentinel2.py` | Digital Earth Africa STAC (AWS S3) | `data/raw/sentinel2/*.nc` |
| `usgs.py` | USGS Africa Mineral GDB + Copperbelt DB | `data/processed/vectors/*.geojson` |
| `ni43101.py` | NI 43-101 PDFs (SEDAR+) | `data/processed/text/*.txt` |

## Layer 2 — Processing (`src/processing/`)

### NLP (`nlp/`)
- PDF text extraction via PyMuPDF with Tesseract OCR fallback
- Tokenization, stopword removal, geological lemmatization
- Custom geological vocabulary for spaCy NER training

### Spectral (`spectral/`)
- Band ratio analysis: clay index (B11/B12), iron oxide index (B11/B8A)
- Composite alteration score normalised to [0, 1]
- Output: GeoTIFF raster at 20m resolution

## Layer 3 — Intelligence (`src/intelligence/`)

### NER (`ner/`)
- Custom spaCy model trained on geological entities:
  `MINERAL`, `DEPOSIT_TYPE`, `COORDINATE`, `ORE_GRADE`, `DEPTH`
- Training data: annotated NI 43-101 report excerpts

### Topic modeling (`topic_modeling/`)
- LDA (Latent Dirichlet Allocation) via Gensim
- 15 topics across 100+ reports
- Surfaces patterns: deposit geology type, drill methodology, grade distribution

### Scoring (`scoring/`)
- Gradient Boosting (LightGBM) classifier
- Features: NER-extracted attributes + spectral alteration score + spatial proximity
- Training labels: known USGS copper deposit locations (positive) vs random non-deposit points (negative)
- Output: site viability score 0–1 per grid cell

## Layer 4 — Platform (`src/platform/`)

### API (`api/`)
- FastAPI backend
- Endpoints: `/predict`, `/sites`, `/reports`, `/health`
- Auth: Azure AD JWT validation
- Async, Pydantic-validated request/response models

### Dashboard (`dashboard/`)
- React + TypeScript frontend
- Mapbox GL JS heatmap of scored sites
- Ranked site list with export to PDF/CSV
- Role-based views: geologist, executive, admin

## Data flow

```
NI 43-101 PDFs ──────────────────────────────────────┐
                                                      ▼
                                              Custom NER ──────────────────┐
                                              Topic LDA ──────────────────┐│
                                                                          ││
Sentinel-2 B11/B12/B8A ──→ Band ratios ──→ Alteration score ────────────┐││
                                                                         │││
USGS deposit GDB ──→ Known deposit coords (training labels) ────────────┐│││
                                                                        ││││
                                                                        ▼▼▼▼
                                                              GBM Scorer (0–1)
                                                                        │
                                                                        ▼
                                                         Ranked site heatmap GeoJSON
                                                                        │
                                                                        ▼
                                                    FastAPI → React dashboard / Enterprise API
```

## Cloud deployment (Azure)

| Component | Azure Service |
|-----------|--------------|
| Raw data storage | Azure Data Lake Gen2 |
| Model training | Azure ML |
| Model serving | Azure ML Online Endpoints |
| API hosting | Azure Container Apps |
| Frontend hosting | Azure Static Web Apps |
| Auth | Azure Active Directory |
| Monitoring | Azure Monitor + Application Insights |
