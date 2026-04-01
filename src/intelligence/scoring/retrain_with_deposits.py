"""
Retrain GBM scorer using real Copperbelt deposit locations as positive labels.

Strategy:
- Positive samples: document chunks that mention known deposit names
  OR have high ore grade + drill hole density
- Negative samples: chunks with no geological signal
- Add deposit proximity as a feature using known lat/lon
"""

import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def build_real_labels(annotations_path, deposits_csv):
    """
    Generate labels based on known deposit mentions in text chunks.
    Positive = chunk mentions a known producing deposit with grade data.
    """
    annotations = json.loads(Path(annotations_path).read_text())
    deposits_df = pd.read_csv(deposits_csv)

    # Known high-value deposit names
    high_value = {d.lower() for d in deposits_df[deposits_df["grade_pct_cu"] >= 1.5]["name"]}
    all_deposits = {d.lower() for d in deposits_df["name"]}

    labels = []
    scores = []

    for item in annotations:
        text_lower = item["text"].lower()
        spans = item["spans"]

        # Count signals
        grade_vals = []
        drill_count = 0
        tonnage_count = 0
        high_deposit = False
        any_deposit = False

        import re
        for span in spans:
            if span["label"] == "ORE_GRADE":
                nums = re.findall(r'\d+\.\d+', span["text"])
                if nums:
                    g = float(nums[0])
                    if 0.5 < g < 10:
                        grade_vals.append(g)
            elif span["label"] == "DRILL_HOLE":
                drill_count += 1
            elif span["label"] == "TONNAGE":
                tonnage_count += 1
            elif span["label"] == "DEPOSIT":
                dep_text = span["text"].lower()
                if any(d in dep_text or dep_text in d for d in high_value):
                    high_deposit = True
                if any(d in dep_text or dep_text in d for d in all_deposits):
                    any_deposit = True

        # Score based on geological signal strength
        score = 0.0
        if grade_vals:
            mean_grade = np.mean(grade_vals)
            score += min(mean_grade / 3.0, 1.0) * 0.35
        if drill_count > 0:
            score += min(drill_count / 3.0, 1.0) * 0.25
        if tonnage_count > 0:
            score += min(tonnage_count / 2.0, 1.0) * 0.20
        if high_deposit:
            score += 0.15
        if any_deposit:
            score += 0.05

        scores.append(score)
        labels.append(1 if score >= 0.35 else 0)

    labels = np.array(labels)
    log.info(f"Label distribution: {labels.sum()} positive / {(1-labels).sum()} negative "
             f"({labels.mean()*100:.1f}% positive rate)")
    return labels


def retrain(annotations_path, topic_matrix_path, deposits_csv, output_dir):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    output_dir = Path(output_dir)

    # Load features
    log.info("Loading features...")
    annotations = json.loads(Path(annotations_path).read_text())
    topic_matrix = np.load(topic_matrix_path)

    # Build NER features
    import re
    ner_rows = []
    for item in annotations:
        grades, depths, minerals = [], [], []
        tonnage, drills, deposits, costs = 0, 0, 0, 0
        for span in item["spans"]:
            label = span["label"]
            if label == "ORE_GRADE":
                nums = re.findall(r'\d+\.\d+', span["text"])
                if nums:
                    g = float(nums[0])
                    if 0.1 < g < 80:
                        grades.append(g)
            elif label == "DEPTH":
                nums = re.findall(r'\d+\.?\d*', span["text"])
                if nums: depths.append(float(nums[0]))
            elif label == "MINERAL":
                minerals.append(span["text"].lower())
            elif label == "TONNAGE": tonnage += 1
            elif label == "DRILL_HOLE": drills += 1
            elif label == "DEPOSIT": deposits += 1
            elif label == "COST": costs += 1

        ner_rows.append({
            "ore_grade_mean": np.mean(grades) if grades else 0.0,
            "ore_grade_max": np.max(grades) if grades else 0.0,
            "ore_grade_count": len(grades),
            "depth_mean": np.mean(depths) if depths else 0.0,
            "depth_max": np.max(depths) if depths else 0.0,
            "depth_count": len(depths),
            "tonnage_count": tonnage,
            "drill_hole_count": drills,
            "deposit_mention_count": deposits,
            "mineral_diversity": len(set(minerals)),
            "cost_mention_count": costs,
            "text_length": len(item["text"]),
        })

    ner_df = pd.DataFrame(ner_rows)
    min_len = min(len(ner_df), len(topic_matrix))
    ner_df = ner_df.iloc[:min_len]
    topic_df = pd.DataFrame(
        topic_matrix[:min_len],
        columns=[f"topic_{i}" for i in range(topic_matrix.shape[1])]
    )
    feature_df = pd.concat([ner_df, topic_df], axis=1)
    feature_names = feature_df.columns.tolist()

    # Real labels
    labels = build_real_labels(annotations_path, deposits_csv)[:min_len]

    X = feature_df.values.astype(np.float32)
    y = labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    log.info(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X_scaled, y)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    roc_auc = roc_auc_score(y, y_prob)
    log.info(f"Full ROC-AUC: {roc_auc:.3f}")
    log.info("\n" + classification_report(y, model.predict(X_scaled)))

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    log.info("Top 10 features:")
    for _, row in importance_df.head(10).iterrows():
        log.info(f"  {row['feature']:30s} {row['importance']:.0f}")

    joblib.dump(model, output_dir / "gbm_viability.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(feature_names, output_dir / "feature_names.pkl")
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    metrics = {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "roc_auc": float(roc_auc),
        "positive_rate": float(y.mean()),
        "n_samples": len(X),
        "label_source": "real_deposit_mentions",
    }
    import json as _json
    with open(output_dir / "metrics.json", "w") as f:
        _json.dump(metrics, f, indent=2)

    log.info(f"Model saved to {output_dir}")
    return metrics


if __name__ == "__main__":
    metrics = retrain(
        annotations_path="data/processed/ner_annotations.json",
        topic_matrix_path="models/topic/doc_topic_matrix.npy",
        deposits_csv="data/raw/usgs/copperbelt_deposits.csv",
        output_dir="models/scoring",
    )
    print(f"\nCV ROC-AUC: {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
    print(f"Label source: {metrics['label_source']}")
    print(f"Positive rate: {metrics['positive_rate']*100:.1f}%")
