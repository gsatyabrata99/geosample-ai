"""
Auto-annotator for geological NER training data.

Uses regex patterns to create silver-standard annotations
from the Kamoa-Kakula technical report text.

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


PATTERNS = {
    "ORE_GRADE": [
        r'\d+\.\d+\s*%\s*(?:Cu|copper|TCu|total copper)',
        r'grade\s+of\s+\d+\.\d+\s*%',
        r'grading\s+\d+\.\d+\s*%',
        r'cut[- ]off\s+grade\s+of\s+\d+\.?\d*\s*%',
        r'\d+\.\d+\s*%\s*total\s+copper',
        r'grade\s+of\s+\d+\.\d+\s*percent',
        r'\d+\.\d+\s*percent\s+(?:Cu|copper)',
    ],
    "TONNAGE": [
        r'\d[\d,]*(?:\.\d+)?\s*billion\s+tonnes?(?:\s+of\s+(?:ore|copper))?',
        r'\d[\d,]*(?:\.\d+)?\s*million\s+tonnes?(?:\s+of\s+(?:ore|copper))?',
        r'\d[\d,]*(?:\.\d+)?\s*million\s+metric\s+tonnes?',
        r'\d[\d,]*(?:\.\d+)?\s*(?:Mt|kt)\b(?!\s*(?:%|Cu|copper))',
        r'\d[\d,]*(?:\.\d+)?\s*Mtpa\b',
        r'\d[\d,]*(?:\.\d+)?\s*million\s+tonne(?:\s+per\s+annum)?',
        r'(?:approximately|about|over|around)\s+\d[\d,]*(?:\.\d+)?\s*(?:million|billion)\s+tonnes?',
        r'\d[\d,]*(?:\.\d+)?\s*tonnes?\s+per\s+annum',
        r'\d[\d,]*(?:\.\d+)?\s*tpa\b',
    ],
    "DEPOSIT": [
        r'Kamoa[–-]Kakula',
        r'Kamoa\s+[1-6]',
        r'Kakula\s+West',
        r'Kakula\s+North',
        r'Kakula\s+South',
        r'Kansoko\s+Sud',
        r'Kansoko\s+Nord',
        r'Kansoko',
        r'Kakula',
        r'Makoko\s+(?:North|South)',
        r'Makoko',
        r'Kirumba',
        r'Kamoa',
    ],
    "DRILL_HOLE": [
        r'\bDD\d{3,4}[A-Z]?\b',
        r'\bKCD[-_]\d+[A-Z]?\b',
        r'\bKMD[-_]\d+[A-Z]?\b',
    ],
    "DEPTH": [
        r'\b\d+(?:\.\d+)?\s*m\b(?!\s*(?:%|Cu|copper|tpa|Mtpa))',
        r'depth\s+of\s+\d+(?:\.\d+)?\s*(?:m|metres?|meters?)',
        r'\d+(?:\.\d+)?\s*metres?\s+(?:below|depth|deep|thick|thickness)',
        r'(?:between|from)\s+\d+(?:\.\d+)?\s*m\s+(?:and|to)\s+\d+(?:\.\d+)?\s*m',
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
        r'\bcarrollite\b',
        r'\bcubanite\b',
    ],
    "LOCATION": [
        r'\bDRC\b',
        r'Democratic\s+Republic\s+of\s+(?:the\s+)?Congo',
        r'\bZambia\b',
        r'\bKolwezi\b',
        r'\bKatanga\b',
        r'\bLubumbashi\b',
        r'\bLualaba\b',
    ],
    "COST": [
        r'\$\d+(?:\.\d+)?\/lb\.?',
        r'\$\d+(?:\.\d+)?\s+per\s+(?:pound|tonne|lb)',
        r'C1\s+(?:cost\s+)?of\s+\$\d+(?:\.\d+)?',
        r'cash\s+cost.*?\$\d+(?:\.\d+)?\/lb',
    ],
}


def _clean_for_annotation(text: str) -> str:
    """
    Remove newlines and extra whitespace within text before annotation.
    This prevents span misalignment caused by mid-entity newlines.
    """
    # Replace newlines with space
    text = re.sub(r'\n+', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences."""
    # First clean newlines
    text = _clean_for_annotation(text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def annotate_text(text: str) -> list[dict]:
    """
    Find all entity matches and return spaCy-format annotations.
    Text is cleaned (newlines removed) before annotation to prevent
    span alignment issues.
    """
    sentences = _split_sentences(text)
    annotated = []

    for sent in sentences:
        # Sentence is already newline-free from _split_sentences
        entities = []
        for label, patterns in PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, sent, re.IGNORECASE):
                    start, end = match.start(), match.end()
                    # Skip if matched text contains newlines (safety check)
                    if '\n' in sent[start:end]:
                        continue
                    # Skip if overlaps with existing entity
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


def save_training_data(annotations: list[dict], output_path: str | Path) -> None:
    """Save annotations in spaCy v3 JSON training format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
