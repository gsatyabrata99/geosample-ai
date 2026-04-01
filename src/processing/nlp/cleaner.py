"""
Geological text cleaner for NI 43-101 reports.
Handles noisy PDF extraction output — page headers, footers,
table artifacts, and non-geological boilerplate.
"""

import re
from pathlib import Path


# Stopwords specific to mining/geology reports to KEEP (override general stopwords)
GEO_KEEP = {
    "copper", "grade", "ore", "deposit", "drill", "hole", "depth",
    "intercept", "assay", "mineralization", "alteration", "lithology",
    "indicated", "inferred", "measured", "reserve", "resource",
    "tonne", "grade", "cut-off", "stope", "pillar", "shaft",
    "collar", "azimuth", "dip", "strike", "footwall", "hangingwall",
    "chalcopyrite", "malachite", "bornite", "chalcocite",
}

# Boilerplate patterns to strip
STRIP_PATTERNS = [
    r'(?i)forward[- ]looking statements?.*?(?=\n\n|\Z)',  # legal boilerplate
    r'(?i)this\s+(?:report|document)\s+has\s+been\s+prepared.*?(?=\n\n|\Z)',
    r'(?i)qualified\s+person[s]?\s+disclosure.*?(?=\n\n|\Z)',
    r'Page\s+\d+\s+of\s+\d+',           # page numbers
    r'CONFIDENTIAL.*?\n',                 # confidentiality headers
    r'(?m)^\s*\d+\s*$',                  # standalone page numbers
    r'©.*?\n',                            # copyright lines
    r'(?m)^={3,}$',                       # separator lines
    r'(?m)^-{3,}$',
]


def clean_geological_text(text: str) -> str:
    """
    Clean extracted PDF text for NLP processing.

    Steps:
    1. Strip legal boilerplate and page artifacts
    2. Normalise whitespace
    3. Fix common PDF extraction artifacts
    4. Preserve geological entities and numbers
    """
    # Fix common PDF ligature artifacts
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    text = text.replace("\x0c", "\n")  # form feed → newline

    # Strip boilerplate
    for pattern in STRIP_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    # Normalise hyphenation from line breaks (e.g. "mineral-\nization" → "mineralization")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Fix grade formatting: "2 .65%" → "2.65%", "2. 65%" → "2.65%"
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)

    # Standardise units: "metres" → "m", keep both for NER context
    text = re.sub(r'\bmetres\b', 'metres', text)  # keep as-is, NER handles both

    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Split geological text into sentences.
    Handles abbreviations common in mining reports (e.g., "Cu.", "m.", "Mt.").
    """
    # Protect common abbreviations from sentence splitting
    protected = [
        (r'\bCu\.', 'Cu_ABBR'),
        (r'\bPb\.', 'Pb_ABBR'),
        (r'\bZn\.', 'Zn_ABBR'),
        (r'\bAu\.', 'Au_ABBR'),
        (r'\bAg\.', 'Ag_ABBR'),
        (r'\bMt\.', 'Mt_ABBR'),
        (r'\bkt\.', 'kt_ABBR'),
        (r'\bFig\.', 'Fig_ABBR'),
        (r'\bvs\.', 'vs_ABBR'),
        (r'\betc\.', 'etc_ABBR'),
        (r'\bapprox\.', 'approx_ABBR'),
        (r'\bref\.', 'ref_ABBR'),
    ]

    for pattern, replacement in protected:
        text = re.sub(pattern, replacement, text)

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore abbreviations
    restored = []
    for sent in sentences:
        for pattern, replacement in protected:
            sent = sent.replace(replacement, pattern.replace(r'\.', '.').replace(r'\b', ''))
        restored.append(sent.strip())

    return [s for s in restored if len(s) > 20]


def process_file(input_path: str | Path, output_path: str | Path) -> dict:
    """
    Clean a single extracted text file and return stats.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = input_path.read_text(encoding='utf-8')
    cleaned = clean_geological_text(raw)
    sentences = split_into_sentences(cleaned)

    output_path.write_text(cleaned, encoding='utf-8')

    return {
        'input_chars': len(raw),
        'output_chars': len(cleaned),
        'sentences': len(sentences),
        'reduction_pct': round((1 - len(cleaned) / len(raw)) * 100, 1),
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    input_dir = Path("data/processed/text")
    output_dir = Path("data/processed/text_clean")

    for txt_file in input_dir.glob("*.txt"):
        out_file = output_dir / txt_file.name
        stats = process_file(txt_file, out_file)
        log.info(
            f"{txt_file.name}: {stats['input_chars']:,} → {stats['output_chars']:,} chars "
            f"({stats['reduction_pct']}% reduction), {stats['sentences']} sentences"
        )
