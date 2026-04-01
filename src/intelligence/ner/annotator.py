"""
Auto-annotator for geological NER training data.

Uses regex patterns to create silver-standard annotations
from the Kamoa-Kakula technical report text.
These are used as the starting point for NER model training.

Entity labels:
  ORE_GRADE     — "2.65% Cu", "grade of 2.82%"
  TONNAGE       — "466 million tonnes", "1.27 billion tonnes"
  DEPOSIT       — "Kamoa 1", "Kakula West", "Kansoko Sud"
  DRILL_HOLE    — "DD996", "DD1080"
  DEPTH         — "450m", "depth of 320 metres"
  MINERAL       — "chalcopyrite", "malachite", "bornite"
  LOCATION      — "DRC", "Zambia", "Kolwezi", "Katanga"
  COST          — "$2.60/lb", "$4.50/lb"
"""

import json
import re
from pathlib import Path


# ── Entity patterns ───────────────────────────────────────────────────────────

PATTERNS = {
    "ORE_GRADE": [
        r'\d+\.\d+\s*%\s*(?:Cu|copper|TCu|total copper)',
        r'grade\s+of\s+\d+\.\d+\s*%',
        r'grading\s+\d+\.\d+\s*%',
        r'cut[- ]off\s+grade\s+of\s+\d+\.?\d*\s*%',
        r'\d+\.\d+\s*%\s*total\s+copper',
    ],
    "TONNAGE": [
        r'\d[\d,\.]*\s*(?:billion|million)\s+tonnes?(?:\s+of\s+(?:ore|copper))?',
        r'\d[\d,\.]*\s*(?:Mt|kt|Mt\s+of\s+ore)',
        r'\d[\d,\.]*\s*million\s+metric\s+tonnes?',
    ],
    "DEPOSIT": [
        r'Kamoa[- ]Kakula',
        r'Kamoa\s*[1-6]',
        r'Kakula\s*(?:West|North|South)?',
        r'Kansoko\s*(?:Sud|Nord)?',
        r'Kakula\s+West',
        r'Makoko\s*(?:North|South)?',
        r'Kirumba',
    ],
    "DRILL_HOLE": [
        r'\bDD\d{3,4}[A-Z]?\b',
        r'\bKCD[-_]\d+[A-Z]?\b',
        r'\bKMD[-_]\d+[A-Z]?\b',
    ],
    "DEPTH": [
        r'\d+\.?\d*\s*m\b(?!\s*%)',
        r'depth\s+of\s+\d+\.?\d*\s*(?:m|metres?|meters?)',
        r'\d+\.?\d*\s*metres?\s+(?:below|depth|deep)',
    ],
    "MINERAL": [
        r'\bchalcopyrite\b',
        r'\bmalachite\b',
        r'\bbornite\b',
        r'\bchalcocite\b',
        r'\bcuprite\b',
        r'\bazurite\b',
        r'\bcovellite\b',
        r'\bdigenite\b',
        r'\bpyrite\b',
        r'\bpyrrhotite\b',
        r'\bcobaltite\b',
    ],
    "LOCATION": [
        r'\bDRC\b',
        r'\bDemocratic Republic of (?:the )?Congo\b',
        r'\bZambia\b',
        r'\bKolwezi\b',
        r'\bKatanga\b',
        r'\bLubumbashi\b',
        r'\bKopje\b',
    ],
    "COST": [
        r'\$\d+\.?\d*\/lb\.?',
        r'\$\d+\.?\d*\s+per\s+(?:pound|tonne|lb)',
        r'C1\s+(?:cost\s+)?of\s+\$\d+\.?\d*',
    ],
}


def annotate_text(text: str) -> list[dict]:
    """
    Find all entity matches in text and return spaCy-format annotations.

    Returns list of:
    {
        "text": "sentence text",
        "entities": [(start, end, label), ...]
    }
    """
    sentences = _split_sentences(text)
    annotated = []

    for sent in sentences:
        entities = []
        for label, patterns in PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, sent, re.IGNORECASE):
                    start, end = match.start(), match.end()
                    # Check for overlaps with existing entities
                    overlap = any(
                        not (end <= e[0] or start >= e[1])
                        for e in entities
                    )
                    if not overlap:
                        entities.append((start, end, label))

        if entities:
            annotated.append({
                "text": sent,
                "entities": sorted(entities, key=lambda x: x[0])
            })

    return annotated


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter for geological text."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def save_training_data(annotations: list[dict], output_path: str | Path) -> None:
    """Save annotations in spaCy v3 JSON training format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # spaCy training format
    training_data = []
    for item in annotations:
        training_data.append({
            "text": item["text"],
            "spans": [
                {
                    "start": e[0],
                    "end": e[1],
                    "label": e[2],
                    "text": item["text"][e[0]:e[1]]
                }
                for e in item["entities"]
            ]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)


def print_sample(annotations: list[dict], n: int = 10) -> None:
    """Print sample annotations for inspection."""
    print(f"\nTotal annotated sentences: {len(annotations)}")
    entity_counts = {}
    for item in annotations:
        for _, _, label in item["entities"]:
            entity_counts[label] = entity_counts.get(label, 0) + 1

    print("\nEntity counts:")
    for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:15} {count:4d}")

    print(f"\nSample annotations (first {n}):")
    for item in annotations[:n]:
        print(f"\n  TEXT: {item['text'][:100]}...")
        for start, end, label in item["entities"]:
            print(f"    [{label}] '{item['text'][start:end]}'")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    text_dir = Path("data/processed/text")
    txt_files = list(text_dir.glob("*.txt"))

    if not txt_files:
        print("No text files found. Run src/ingestion/ni43101.py first.")
        exit(1)

    all_annotations = []
    for txt_file in txt_files:
        print(f"Annotating {txt_file.name}...")
        text = txt_file.read_text(encoding="utf-8")
        annotations = annotate_text(text)
        all_annotations.extend(annotations)

    print_sample(all_annotations, n=5)

    output_path = Path("data/processed/ner_annotations.json")
    save_training_data(all_annotations, output_path)
    print(f"\nSaved {len(all_annotations)} annotated sentences to {output_path}")
