"""
Train a custom spaCy NER model on geological entities
from the auto-annotated Kamoa-Kakula technical report data.
"""

import json
import random
import logging
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_annotations(json_path: str | Path) -> list[tuple]:
    """Load annotations and convert to spaCy training format."""
    data = json.loads(Path(json_path).read_text())
    training = []
    skipped = 0
    for item in data:
        text = item["text"]
        entities = [(s["start"], s["end"], s["label"]) for s in item["spans"]]
        # Validate spans don't exceed text length
        valid = [(s, e, l) for s, e, l in entities if e <= len(text) and s >= 0]
        if len(valid) != len(entities):
            skipped += len(entities) - len(valid)
        if valid:
            training.append((text, {"entities": valid}))
    if skipped:
        log.warning(f"Skipped {skipped} invalid spans")
    return training


def train_ner(
    annotations_path: str | Path,
    output_dir: str | Path,
    n_iter: int = 30,
    test_split: float = 0.2,
    dropout: float = 0.3,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading annotations...")
    data = load_annotations(annotations_path)
    random.shuffle(data)

    split = int(len(data) * (1 - test_split))
    train_data = data[:split]
    test_data = data[split:]
    log.info(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # Build pipeline from blank English model
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)

    # Add all entity labels
    for _, annotations in train_data:
        for _, _, label in annotations["entities"]:
            ner.add_label(label)

    log.info(f"Labels: {ner.labels}")

    # Training loop
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    best_f1 = 0.0

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()

        for iteration in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(examples, drop=dropout, losses=losses)

            # Evaluate every 5 iterations
            if (iteration + 1) % 5 == 0:
                scores = evaluate(nlp, test_data)
                f1 = scores["f1"]
                log.info(
                    f"Iter {iteration+1:3d} | loss: {losses['ner']:.3f} | "
                    f"P: {scores['precision']:.3f} | R: {scores['recall']:.3f} | F1: {f1:.3f}"
                )
                if f1 > best_f1:
                    best_f1 = f1
                    nlp.to_disk(output_dir / "best_model")
                    log.info(f"  → New best model saved (F1={f1:.3f})")

    # Save final model
    nlp.to_disk(output_dir / "final_model")
    log.info(f"\nTraining complete. Best F1: {best_f1:.3f}")
    log.info(f"Models saved to {output_dir}")

    # Final evaluation with per-label breakdown
    log.info("\nPer-label scores on test set:")
    scores = evaluate_per_label(nlp, test_data)
    for label, s in sorted(scores.items()):
        log.info(f"  {label:15} P:{s['p']:.2f}  R:{s['r']:.2f}  F1:{s['f1']:.2f}  ({s['support']} examples)")


def evaluate(nlp, test_data: list) -> dict:
    examples = []
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    scores = nlp.evaluate(examples)
    return {
        "precision": scores["ents_p"],
        "recall": scores["ents_r"],
        "f1": scores["ents_f"],
    }


def evaluate_per_label(nlp, test_data: list) -> dict:
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for text, annotations in test_data:
        doc = nlp(text)
        pred = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        gold = {(s, e, l) for s, e, l in annotations["entities"]}
        for ent in pred:
            if ent in gold:
                tp[ent[2]] += 1
            else:
                fp[ent[2]] += 1
        for ent in gold:
            if ent not in pred:
                fn[ent[2]] += 1

    results = {}
    all_labels = set(list(tp.keys()) + list(fn.keys()))
    for label in all_labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[label] = {"p": p, "r": r, "f1": f1, "support": tp[label] + fn[label]}
    return results


if __name__ == "__main__":
    train_ner(
        annotations_path="data/processed/ner_annotations.json",
        output_dir="models/ner",
        n_iter=30,
    )
