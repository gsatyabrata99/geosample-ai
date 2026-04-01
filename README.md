# GeoSample AI

**AI-guided copper exploration sampling optimizer for Central/West Africa.**

GeoSample AI ingests unstructured geological data — NI 43-101 technical reports, drill logs, and Sentinel-2 satellite imagery — and outputs a ranked heatmap of optimal copper sampling sites. Built for mineral exploration teams who need to prioritize ground resources before a single drill bit turns.

![GeoSample AI Dashboard](docs/dashboard_preview.png)

---

## What it does

A mining team uploads a technical report or pastes drill intercept text. GeoSample AI:

1. **Extracts geological entities** — ore grades, deposit names, drill holes, depths, minerals, tonnages — using a custom spaCy NER model trained on 608 pages of real NI 43-101 data
2. **Identifies geological themes** — via LDA topic modeling across 15 topics (Drilling & Geology, Mine Development, Metallurgical Testwork, Smelting, Environment, etc.)
3. **Scores site viability** — LightGBM classifier combining 27 NER + LDA features, trained on 20 known Copperbelt copper deposits
4. **Visualizes on a real map** — Mapbox dark map with scored site bubbles and a Sentinel-2 copper alteration heatmap overlay

The Kamoa-Kakula 2026 Mineral Reserve and Resource Technical Report (the world's highest-grade copper development, published March 31 2026) scores **99.9% viability**.

---

## Architecture
```
NI 43-101 PDFs          Sentinel-2 Imagery       USGS Deposit Data
      │                        │                        │
      ▼                        ▼                        ▼
 PDF Extractor           Band Analysis           Deposit Registry
 (PyMuPDF + OCR)        (B11/B12/B8A SWIR)      (20 known sites)
      │                        │                        │
      └──────────────┬──────────┘                       │
                     ▼                                  │
              Text Cleaner                              │
                     │                                  │
          ┌──────────┴──────────┐                       │
          ▼                     ▼                       │
      NER Model            LDA Model                    │
   (spaCy, 8 labels)    (Gensim, 15 topics)             │
          │                     │                       │
          └──────────┬──────────┘                       │
                     ▼                                  │
              Feature Matrix                            │
              (27 features)           ◄─────────────────┘
                     │
                     ▼
           LightGBM Scorer
           (CV AUC: 0.841)
                     │
                     ▼
              FastAPI Backend ──► React Dashboard
              (6 endpoints)         (Mapbox + heatmap)
```

---

## NER Model Performance

Trained on Kamoa-Kakula 2026 MRE (608 pages, 1.17M characters):

| Entity | F1 | Examples |
|---|---|---|
| DEPOSIT | 0.98 | Kamoa-Kakula, Kakula West, Kansoko Sud |
| MINERAL | 1.00 | chalcopyrite, bornite, chalcocite |
| LOCATION | 1.00 | DRC, Zambia, Kolwezi, Lualaba |
| DRILL_HOLE | 0.92 | DD1080, DD1724 |
| DEPTH | 0.89 | 450m, 12.5m |
| ORE_GRADE | 0.88 | 3.94% Cu, 2.53% TCu |
| TONNAGE | 0.82 | 523 million tonnes, 17 Mtpa |
| COST | 1.00 | $2.60/lb |

---

## Data Sources

| Source | Description | Access |
|---|---|---|
| Kamoa-Kakula 2026 MRE | 608-page NI 43-101 technical report, March 31 2026 | Free via SEDAR+ |
| Digital Earth Africa STAC | Sentinel-2 L2A imagery, bands B11/B12/B8A | Free, no auth |
| USGS Copperbelt Deposits | 20 known DRC/Zambia copper deposits | Curated from literature |

---

## Stack

**ML/Data:** Python, spaCy, Gensim, LightGBM, scikit-learn, SHAP, rasterio, odc-stac, rioxarray

**Backend:** FastAPI, Uvicorn, PyMuPDF

**Frontend:** React, Mapbox GL, react-map-gl, Vite, Axios

**Infrastructure:** GitHub Actions CI (ruff + pytest), pre-commit hooks

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Mapbox account (free tier)

### Backend
```bash
git clone https://github.com/gsatyabrata99/geosample-ai
cd geosample-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train models (or skip — pre-trained weights included for config/metadata)
python src/intelligence/ner/annotator.py
python src/intelligence/ner/train.py
python src/intelligence/topic_modeling/lda.py
python src/intelligence/scoring/retrain_with_deposits.py

# Start API
python -m uvicorn src.platform.api.main:app --reload --port 8000
```

### Frontend
```bash
cd src/platform/dashboard
npm install
cp .env.example .env
# Add your Mapbox token to .env
npm run dev
```

Open `http://localhost:5173`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Model status (NER/GBM/LDA) |
| POST | `/predict` | Score geological text (~6ms) |
| POST | `/predict/batch` | Score multiple text blocks |
| POST | `/reports/analyze` | Upload and analyze PDF report |
| GET | `/sites` | Ranked site list with scores |
| GET | `/model/info` | NER labels, feature importance |

### Example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Drill hole DD1080 intersected 12.5m at 3.94% Cu from 450m depth in the Kakula deposit. Chalcopyrite and bornite are the primary ore minerals. The indicated resource is 523 million tonnes at 2.53% Cu."
  }'
```

Returns:
```json
{
  "viability_score": 0.9991,
  "entities": [
    {"text": "DD1080", "label": "DRILL_HOLE"},
    {"text": "3.94% Cu", "label": "ORE_GRADE"},
    {"text": "450m", "label": "DEPTH"},
    {"text": "Kakula", "label": "DEPOSIT"},
    {"text": "chalcopyrite", "label": "MINERAL"},
    {"text": "523 million tonnes", "label": "TONNAGE"}
  ],
  "top_topics": [
    {"label": "Drilling & Geology", "probability": 0.561},
    {"label": "Mine Development", "probability": 0.400}
  ],
  "processing_time_ms": 6.59
}
```

---

## Results

- **Kamoa-Kakula 2026 MRE** → 99.9% viability (world's highest-grade copper development)
- **NER overall F1** → 0.994 on test set
- **LDA coherence** → 0.61 (c_v), 15 topics
- **GBM CV AUC** → 0.841 (trained on real Copperbelt deposit locations)
- **API latency** → ~6ms per prediction

---

## Project Structure
```
geosample-ai/
├── src/
│   ├── ingestion/          # Sentinel-2, USGS, NI 43-101 PDF loaders
│   ├── processing/         # Text cleaning, spectral band analysis
│   ├── intelligence/       # NER, LDA, GBM scorer
│   └── platform/
│       ├── api/            # FastAPI backend
│       └── dashboard/      # React frontend
├── models/
│   ├── ner/                # Trained spaCy NER model
│   ├── topic/              # LDA model + document-topic matrix
│   └── scoring/            # LightGBM + scaler + feature importance
├── data/
│   ├── raw/                # Source data (gitignored)
│   └── processed/          # NER annotations, cleaned text
├── configs/                # Pipeline parameters
└── scripts/                # Download and run scripts
```

---

## Academic Context

Built as the capstone project for CIS 8045 (Advanced Data Analytics) at Georgia State University, MS Information Systems (AI & Data-Driven Analytics), Spring 2026.

---

## Author

**Ganesh Satyabrata** — AI/ML Product Manager  
[github.com/gsatyabrata99](https://github.com/gsatyabrata99) · [linkedin.com/in/ganeshs99](https://linkedin.com/in/ganeshs99)
