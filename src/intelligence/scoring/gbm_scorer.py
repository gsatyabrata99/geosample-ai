"""
GBM site viability scorer for copper mineral exploration.

Combines features from:
  1. NER extractions (ore grades, deposit mentions, drill hole density)
  2. LDA topic distributions (geological themes per document chunk)
  3. Sentinel-2 spectral alteration scores (when available)
  4. USGS deposit proximity (spatial features)

Training labels: known USGS copper deposit locations (positive)
vs random non-deposit points in same region (negative).

Output: 0-1 viability score per grid cell.
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ── Feature extraction from NER annotations ──────────────────────────────────

def extract_ner_features(annotations_path: str | Path) -> pd.DataFrame:
    """
    Extract numerical features from NER annotation JSON.

    Features per document:
    - ore_grade_mean, ore_grade_max, ore_grade_count
    - tonnage_count
    - drill_hole_count
    - depth_mean, depth_count
    - deposit_mention_count
    - mineral_diversity (unique mineral types)
    - cost_mention_count
    """
    data = json.loads(Path(annotations_path).read_text())
    rows = []

    for item in data:
        text = item["text"]
        spans = item["spans"]

        grades, depths, tonnages = [], [], []
        drill_holes, deposits, minerals, costs = [], [], [], []

        for span in spans:
            label = span["label"]
            val = span["text"]

            if label == "ORE_GRADE":
                import re
                nums = re.findall(r'\d+\.\d+', val)
                if nums:
                    grade = float(nums[0])
                    if 0.1 < grade < 80:  # filter smelter grades
                        grades.append(grade)

            elif label == "DEPTH":
                import re
                nums = re.findall(r'\d+\.?\d*', val)
                if nums:
                    depths.append(float(nums[0]))

            elif label == "TONNAGE":
                tonnages.append(val)

            elif label == "DRILL_HOLE":
                drill_holes.append(val)

            elif label == "DEPOSIT":
                deposits.append(val)

            elif label == "MINERAL":
                minerals.append(val.lower())

            elif label == "COST":
                costs.append(val)

        rows.append({
            "text_snippet": text[:50],
            "ore_grade_mean": np.mean(grades) if grades else 0.0,
            "ore_grade_max": np.max(grades) if grades else 0.0,
            "ore_grade_count": len(grades),
            "depth_mean": np.mean(depths) if depths else 0.0,
            "depth_max": np.max(depths) if depths else 0.0,
            "depth_count": len(depths),
            "tonnage_count": len(tonnages),
            "drill_hole_count": len(drill_holes),
            "deposit_mention_count": len(deposits),
            "mineral_diversity": len(set(minerals)),
            "cost_mention_count": len(costs),
            "text_length": len(text),
        })

    return pd.DataFrame(rows)


def extract_lda_features(topic_matrix_path: str | Path) -> pd.DataFrame:
    """Load document-topic matrix as features."""
    matrix = np.load(topic_matrix_path)
    cols = [f"topic_{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=cols)


def build_feature_matrix(
    annotations_path: str | Path,
    topic_matrix_path: str | Path,
) -> pd.DataFrame:
    """
    Combine NER and LDA features into a single feature matrix.
    Both must have the same number of rows (document chunks).
    """
    ner_df = extract_ner_features(annotations_path)
    lda_df = extract_lda_features(topic_matrix_path)

    # Align by index (both should be 329 chunks if using same source)
    min_len = min(len(ner_df), len(lda_df))
    ner_df = ner_df.iloc[:min_len].reset_index(drop=True)
    lda_df = lda_df.iloc[:min_len].reset_index(drop=True)

    feature_df = pd.concat([ner_df.drop(columns=["text_snippet"]), lda_df], axis=1)
    log.info(f"Feature matrix: {feature_df.shape[0]} samples × {feature_df.shape[1]} features")
    return feature_df


# ── Synthetic label generation (until USGS spatial data is loaded) ─────────

def generate_synthetic_labels(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Generate proxy labels for training based on geological signal strength.

    High-viability indicators:
    - High ore grade (> 2% Cu typical for sediment-hosted deposits)
    - Multiple drill holes mentioned
    - Mineral diversity (chalcopyrite + bornite = classic Cu assemblage)
    - Strong LDA topic signal for Geology or Drilling topics

    This is replaced by real USGS deposit labels once spatial data is loaded.
    """
    scores = np.zeros(len(feature_df))

    # Grade signal
    scores += np.clip(feature_df["ore_grade_mean"] / 5.0, 0, 1) * 0.30

    # Drill hole density
    scores += np.clip(feature_df["drill_hole_count"] / 10.0, 0, 1) * 0.20

    # Mineral diversity
    scores += np.clip(feature_df["mineral_diversity"] / 5.0, 0, 1) * 0.15

    # Tonnage mentions (resource-defined areas)
    scores += np.clip(feature_df["tonnage_count"] / 5.0, 0, 1) * 0.15

    # Depth mentions (actual drilling data)
    scores += np.clip(feature_df["depth_count"] / 10.0, 0, 1) * 0.10

    # Topic 13 (Drilling/Geology) and Topic 3 (Grade/Recovery)
    if "topic_13" in feature_df.columns:
        scores += feature_df["topic_13"].values * 0.05
    if "topic_3" in feature_df.columns:
        scores += feature_df["topic_3"].values * 0.05

    # Convert to binary labels with threshold
    labels = (scores > 0.25).astype(int)
    log.info(f"Label distribution: {labels.sum()} positive / {(1-labels).sum()} negative "
             f"({labels.mean()*100:.1f}% positive rate)")
    return labels


# ── Model training ────────────────────────────────────────────────────────────

def train_gbm(
    annotations_path: str | Path,
    topic_matrix_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """
    Train a LightGBM site viability scorer.

    Args:
        annotations_path: NER annotations JSON
        topic_matrix_path: document-topic matrix .npy file
        output_dir: where to save model and evaluation results

    Returns:
        Dict with model metrics
    """
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        classification_report, roc_auc_score,
        average_precision_score,
    )
    from sklearn.preprocessing import StandardScaler

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build features
    log.info("Building feature matrix...")
    feature_df = build_feature_matrix(annotations_path, topic_matrix_path)

    # Generate labels
    labels = generate_synthetic_labels(feature_df)

    X = feature_df.values.astype(np.float32)
    y = labels

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train LightGBM
    log.info("Training LightGBM classifier...")
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    log.info(f"Cross-val ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit on all data
    model.fit(X_scaled, y)

    # Full-data evaluation
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    roc_auc = roc_auc_score(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)

    log.info(f"ROC-AUC (full): {roc_auc:.3f}")
    log.info(f"Avg Precision: {avg_precision:.3f}")
    log.info("\nClassification Report:")
    log.info("\n" + classification_report(y, y_pred))

    # Feature importance
    feature_names = feature_df.columns.tolist()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log.info("\nTop 15 features:")
    for _, row in importance_df.head(15).iterrows():
        log.info(f"  {row['feature']:30s} {row['importance']:.4f}")

    # SHAP explanation (sample of 50 docs)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled[:50])
        log.info("SHAP values computed for 50 sample documents")
    except Exception as e:
        log.warning(f"SHAP failed: {e}")

    # Save model and scaler
    joblib.dump(model, output_dir / "gbm_viability.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(feature_names, output_dir / "feature_names.pkl")

    # Save feature importance
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    metrics = {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "roc_auc": float(roc_auc),
        "avg_precision": float(avg_precision),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "positive_rate": float(y.mean()),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"\nModel saved to {output_dir}")
    return metrics


def predict_viability(
    feature_df: pd.DataFrame,
    model_dir: str | Path,
) -> np.ndarray:
    """
    Score new geological text chunks using the trained model.

    Returns array of viability scores in [0, 1].
    """
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "gbm_viability.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")

    X = feature_df.values.astype(np.float32)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


if __name__ == "__main__":
    metrics = train_gbm(
        annotations_path="data/processed/ner_annotations.json",
        topic_matrix_path="models/topic/doc_topic_matrix.npy",
        output_dir="models/scoring",
    )
    print(f"\n=== GBM SCORER RESULTS ===")
    print(f"CV ROC-AUC:    {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.3f}")
    print(f"Avg Precision: {metrics['avg_precision']:.3f}")
    print(f"Samples:       {metrics['n_samples']}")
    print(f"Features:      {metrics['n_features']}")
    print(f"Positive rate: {metrics['positive_rate']*100:.1f}%")
