"""
GeoSample AI — FastAPI backend.

Endpoints:
  GET  /health              — service health check
  POST /predict             — score a block of geological text
  POST /predict/batch       — score multiple text blocks
  GET  /sites               — get top-scored sites (stub until spatial data)
  POST /reports/analyze     — analyze an NI 43-101 report text
  GET  /model/info          — model metadata and feature importance
"""

from __future__ import annotations
import numpy as np

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="GeoSample AI",
    description="AI-guided sample location optimizer for copper mineral exploration",
    version="0.1.0",
)

# Allow React dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model state (loaded once at startup) ─────────────────────────────────────

MODEL_DIR = Path("models")
_ner_model = None
_gbm_model = None
_scaler = None
_feature_names = None
_lda_model = None
_dictionary = None


def load_models():
    global _ner_model, _gbm_model, _scaler, _feature_names, _lda_model, _dictionary
    import spacy
    import joblib
    from gensim import corpora, models

    ner_path = MODEL_DIR / "ner/best_model"
    if ner_path.exists():
        _ner_model = spacy.load(str(ner_path))
        log.info("NER model loaded")
    else:
        log.warning(f"NER model not found at {ner_path}")

    gbm_path = MODEL_DIR / "scoring/gbm_viability.pkl"
    if gbm_path.exists():
        _gbm_model = joblib.load(gbm_path)
        _scaler = joblib.load(MODEL_DIR / "scoring/scaler.pkl")
        _feature_names = joblib.load(MODEL_DIR / "scoring/feature_names.pkl")
        log.info("GBM scorer loaded")
    else:
        log.warning(f"GBM model not found at {gbm_path}")

    lda_path = MODEL_DIR / "topic/lda_model"
    dict_path = MODEL_DIR / "topic/dictionary.dict"
    if lda_path.exists():
        _lda_model = models.LdaModel.load(str(lda_path))
        _dictionary = corpora.Dictionary.load(str(dict_path))
        log.info("LDA model loaded")
    else:
        log.warning(f"LDA model not found at {lda_path}")


@app.on_event("startup")
async def startup_event():
    load_models()


# ── Request / Response models ─────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str = Field(..., min_length=10, description="Geological text to analyze")
    source: str | None = Field(None, description="Source document identifier")


class BatchTextInput(BaseModel):
    texts: list[TextInput] = Field(..., max_length=50)


class EntitySpan(BaseModel):
    text: str
    label: str
    start: int
    end: int


class PredictionResponse(BaseModel):
    viability_score: float = Field(..., ge=0.0, le=1.0)
    entities: list[EntitySpan]
    top_topics: list[dict]
    features: dict
    processing_time_ms: float
    source: str | None


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total_processing_time_ms: float


class SiteResponse(BaseModel):
    site_id: str
    latitude: float
    longitude: float
    viability_score: float
    top_entities: list[str]
    dominant_topic: str


class ModelInfoResponse(BaseModel):
    ner_labels: list[str]
    lda_num_topics: int
    gbm_features: list[str]
    top_features: list[dict]


# ── Core prediction logic ─────────────────────────────────────────────────────

def extract_features_from_text(text: str) -> tuple[dict, list[EntitySpan], list[dict]]:
    """
    Run NER + LDA on text and return features, entities, and topics.
    """
    import re
    import numpy as np

    entities = []
    features = {
        "ore_grade_mean": 0.0,
        "ore_grade_max": 0.0,
        "ore_grade_count": 0,
        "depth_mean": 0.0,
        "depth_max": 0.0,
        "depth_count": 0,
        "tonnage_count": 0,
        "drill_hole_count": 0,
        "deposit_mention_count": 0,
        "mineral_diversity": 0,
        "cost_mention_count": 0,
        "text_length": len(text),
    }

    # NER extraction
    if _ner_model:
        doc = _ner_model(text[:10000])  # cap at 10k chars
        grades, depths = [], []
        minerals = set()

        for ent in doc.ents:
            entities.append(EntitySpan(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            ))

            if ent.label_ == "ORE_GRADE":
                nums = re.findall(r'\d+\.\d+', ent.text)
                if nums:
                    g = float(nums[0])
                    if 0.1 < g < 80:
                        grades.append(g)

            elif ent.label_ == "DEPTH":
                nums = re.findall(r'\d+\.?\d*', ent.text)
                if nums:
                    depths.append(float(nums[0]))

            elif ent.label_ == "TONNAGE":
                features["tonnage_count"] += 1

            elif ent.label_ == "DRILL_HOLE":
                features["drill_hole_count"] += 1

            elif ent.label_ == "DEPOSIT":
                features["deposit_mention_count"] += 1

            elif ent.label_ == "MINERAL":
                minerals.add(ent.text.lower())

            elif ent.label_ == "COST":
                features["cost_mention_count"] += 1

        if grades:
            features["ore_grade_mean"] = float(np.mean(grades))
            features["ore_grade_max"] = float(np.max(grades))
            features["ore_grade_count"] = len(grades)

        if depths:
            features["depth_mean"] = float(np.mean(depths))
            features["depth_max"] = float(np.max(depths))
            features["depth_count"] = len(depths)

        features["mineral_diversity"] = len(minerals)

    # LDA topic inference
    topics = []
    topic_features = {f"topic_{i}": 0.0 for i in range(15)}

    if _lda_model and _dictionary:
        import re as re2
        tokens = re2.findall(r'\b[a-z][a-z]{2,}\b', text.lower())
        bow = _dictionary.doc2bow(tokens)
        if bow:
            topic_dist = _lda_model.get_document_topics(bow, minimum_probability=0.0)
            for topic_id, prob in topic_dist:
                topic_features[f"topic_{topic_id}"] = float(prob)

            top_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:3]
            topic_labels = {
                0: "Flotation Circuit", 1: "Ventilation", 2: "Electrical Infrastructure",
                3: "Grade & Recovery", 4: "Mine Economics", 5: "Water & Pumping",
                6: "Environmental Studies", 7: "Social & Environment", 8: "Smelting",
                9: "Geotechnics", 10: "Mine Planning", 11: "Mine Development",
                12: "Metallurgical Testwork", 13: "Drilling & Geology", 14: "Lab Statistics",
            }
            topics = [
                {"topic_id": int(tid), "probability": round(float(prob), 3),
                 "label": topic_labels.get(tid, f"Topic {tid}")}
                for tid, prob in top_topics
            ]

    features.update(topic_features)
    return features, entities, topics


def score_text(text: str) -> tuple[float, dict, list, list]:
    """Run full pipeline and return viability score."""
    import numpy as np

    features, entities, topics = extract_features_from_text(text)

    if _gbm_model and _scaler and _feature_names:
        feat_vector = np.array([[features.get(f, 0.0) for f in _feature_names]], dtype=np.float32)
        feat_scaled = _scaler.transform(feat_vector)
        score = float(_gbm_model.predict_proba(feat_scaled)[0][1])
    else:
        # Fallback: rule-based score if models not loaded
        score = min(1.0, (
            features["ore_grade_mean"] / 10.0 * 0.4 +
            features["drill_hole_count"] / 5.0 * 0.3 +
            features["mineral_diversity"] / 5.0 * 0.3
        ))

    return float(score), {k: float(v) if hasattr(v, "item") else v for k, v in features.items()}, entities, topics


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "ner": _ner_model is not None,
            "gbm": _gbm_model is not None,
            "lda": _lda_model is not None,
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input: TextInput):
    """Score a block of geological text for copper exploration viability."""
    t0 = time.time()

    try:
        score, features, entities, topics = score_text(input.text)
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        viability_score=round(score, 4),
        entities=entities,
        top_topics=topics,
        features={k: round(v, 4) if isinstance(v, float) else v
                  for k, v in features.items()
                  if not k.startswith("topic_")},
        processing_time_ms=round((time.time() - t0) * 1000, 2),
        source=input.source,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input: BatchTextInput):
    """Score multiple text blocks in one request."""
    t0 = time.time


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata and feature importance."""
    ner_labels = list(_ner_model.get_pipe("ner").labels) if _ner_model else []
    lda_topics = _lda_model.num_topics if _lda_model else 0

    top_features = []
    importance_path = Path("models/scoring/feature_importance.csv")
    if importance_path.exists():
        import pandas as pd
        df = pd.read_csv(importance_path)
        top_features = df.head(10).to_dict(orient="records")

    return ModelInfoResponse(
        ner_labels=ner_labels,
        lda_num_topics=lda_topics,
        gbm_features=_feature_names or [],
        top_features=top_features,
    )


@app.get("/sites")
async def get_sites(min_score: float = 0.5, limit: int = 20):
    """Return top-scored sites."""
    sample_sites = [
        {"site_id": "DRC-001", "lat": -10.737, "lon": 25.497, "score": 0.999,
         "deposit": "Kamoa-Kakula 2026 MRE", "grade_pct": 2.86,
         "commodity": "Copper", "country": "DRC", "report": "Ivanhoe Mines NI 43-101 Mar 2026"},
        {"site_id": "TUR-001", "lat": 41.083, "lon": 40.917, "score": 0.995,
         "deposit": "First Quantum Cayeli", "grade_pct": 3.2,
         "commodity": "Copper-Zinc", "country": "Turkey", "report": "FQM Cayeli Tech Report Oct 2025"},
        {"site_id": "DRC-002", "lat": -10.583, "lon": 26.083, "score": 0.982,
         "deposit": "Kazhiba Copper Project", "grade_pct": 2.1,
         "commodity": "Copper", "country": "DRC", "report": "MMA Kazhiba NI 43-101 2020"},
        {"site_id": "ARG-001", "lat": -28.917, "lon": -69.583, "score": 0.965,
         "deposit": "Aldebaran Rio Grande", "grade_pct": 0.45,
         "commodity": "Copper-Gold", "country": "Argentina", "report": "Centauri-Aldebaran NI 43-101 Apr 2026"},
        {"site_id": "CAN-001", "lat": 49.917, "lon": -75.583, "score": 0.309,
         "deposit": "Kenorland Frotet", "grade_pct": 0.0,
         "commodity": "Gold", "country": "Canada", "report": "Sumitomo-Kenorland Frotet NI 43-101 Jan 2026"},
        {"site_id": "CAN-002", "lat": 56.75, "lon": -130.083, "score": 0.116,
         "deposit": "Skeena Eskay Creek", "grade_pct": 0.0,
         "commodity": "Gold-Silver", "country": "Canada", "report": "Skeena Eskay Creek Tech Report Dec 2023"},
        {"site_id": "CAN-003", "lat": 48.583, "lon": -79.083, "score": 0.075,
         "deposit": "Estrades PEA", "grade_pct": 0.0,
         "commodity": "Zinc-Gold", "country": "Canada", "report": "Estrades PEA 2026"},
    ]
    filtered = [s for s in sample_sites if s["score"] >= min_score][:limit]
    return {"sites": filtered, "count": len(filtered)}


@app.post("/reports/analyze")
async def analyze_report(file: UploadFile = File(...)):
    """Upload a PDF and get viability analysis."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        import fitz, io
        content = await file.read()
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")

    # Score in chunks and take max — model trained on 500-word chunks
    words = text.split()
    chunk_size = 500
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, min(len(words), 10000), chunk_size)]
    
    chunk_scores = []
    all_entities = []
    best_topics = []
    best_features = {}
    
    for chunk in chunks[:20]:  # max 20 chunks
        s, f, e, t = score_text(chunk)
        chunk_scores.append(s)
        all_entities.extend(e)
        if s > max(chunk_scores[:-1], default=0):
            best_topics = t
            best_features = f
    
    score = float(np.max(chunk_scores)) if chunk_scores else 0.0
    features = best_features
    entities = all_entities
    topics = best_topics

    entity_summary = {}
    for ent in entities:
        entity_summary.setdefault(ent.label, []).append(ent.text)

    return {
        "filename": file.filename,
        "viability_score": round(score, 4),
        "text_length": len(text),
        "entity_summary": {k: list(set(v))[:10] for k, v in entity_summary.items()},
        "top_topics": topics,
        "key_features": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in features.items()
                         if not k.startswith("topic_") and v != 0},
    }
