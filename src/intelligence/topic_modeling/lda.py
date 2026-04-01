"""
LDA topic modeling on NI 43-101 geological reports.

Discovers latent geological themes across report text —
deposit geology types, drilling methodology, grade distributions,
metallurgical processes, infrastructure, and environmental topics.

Output: trained LDA model + topic-document matrix for GBM feature input.
"""

import json
import logging
import re
from pathlib import Path

import nltk
from gensim import corpora, models
from gensim.models import CoherenceModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Geological stopwords (in addition to standard English stopwords) ──────────
GEO_STOPWORDS = {
    "figure", "table", "section", "appendix", "report", "technical",
    "page", "ref", "source", "amc", "pty", "ltd", "inc", "corp",
    "kamoa", "kakula", "ivanhoe", "copper", "kansoko",  # too frequent to distinguish topics
    "mineral", "resource", "reserve", "estimate", "project",
    "also", "would", "used", "using", "use", "based", "may",
    "per", "total", "within", "area", "zone", "site",
    "update", "2025", "2024", "2023", "2022", "2026",
    "december", "march", "january", "february", "april",
}


def build_geological_stopwords() -> set:
    """Combine NLTK English stopwords with geological domain stopwords."""
    from nltk.corpus import stopwords
    english_stops = set(stopwords.words("english"))
    return english_stops | GEO_STOPWORDS


def preprocess_text(text: str, stopwords: set) -> list[str]:
    """
    Tokenize and clean text for LDA.
    - Lowercase
    - Remove numbers-only tokens and short tokens
    - Remove stopwords
    - Keep geological compound terms intact
    """
    # Lowercase and split
    text = text.lower()

    # Preserve important compound terms before tokenization
    compound_terms = {
        "cut off grade": "cutoff_grade",
        "ore grade": "ore_grade",
        "drill hole": "drillhole",
        "mineral resource": "mineral_resource",
        "mineral reserve": "mineral_reserve",
        "open pit": "open_pit",
        "underground mine": "underground_mine",
        "tailings storage": "tailings_storage",
        "flotation circuit": "flotation_circuit",
        "heap leach": "heap_leach",
        "solvent extraction": "solvent_extraction",
        "electrowinning": "electrowinning",
        "grade tonnage": "grade_tonnage",
        "copper concentrate": "copper_concentrate",
        "sulphide ore": "sulphide_ore",
        "hypogene mineralization": "hypogene_mineralization",
        "supergene mineralization": "supergene_mineralization",
    }
    for phrase, replacement in compound_terms.items():
        text = text.replace(phrase, replacement)

    # Tokenize
    tokens = re.findall(r'\b[a-z][a-z_]{2,}\b', text)

    # Filter
    tokens = [
        t for t in tokens
        if t not in stopwords
        and len(t) > 2
        and not t.isdigit()
    ]

    return tokens


def build_corpus(text_dir: str | Path, stopwords: set) -> tuple:
    """
    Load all text files, tokenize, build gensim dictionary and corpus.

    Returns:
        (dictionary, corpus, doc_labels)
    """
    text_dir = Path(text_dir)
    txt_files = sorted(text_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {text_dir}")

    log.info(f"Loading {len(txt_files)} text files...")
    all_docs = []
    doc_labels = []

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")

        # Split into chunks of ~500 words for better topic granularity
        words = text.split()
        chunk_size = 500
        chunks = [
            " ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        for i, chunk in enumerate(chunks):
            tokens = preprocess_text(chunk, stopwords)
            if len(tokens) > 20:  # skip near-empty chunks
                all_docs.append(tokens)
                doc_labels.append(f"{txt_file.stem}_chunk{i}")

    log.info(f"Built {len(all_docs)} document chunks")

    # Build dictionary
    dictionary = corpora.Dictionary(all_docs)

    # Filter extremes: remove very rare and very common words
    dictionary.filter_extremes(no_below=3, no_above=0.85, keep_n=5000)
    log.info(f"Dictionary size: {len(dictionary)} terms after filtering")

    # Build bag-of-words corpus
    corpus = [dictionary.doc2bow(doc) for doc in all_docs]

    return dictionary, corpus, doc_labels, all_docs


def train_lda(
    text_dir: str | Path,
    output_dir: str | Path,
    num_topics: int = 15,
    passes: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Train LDA model on geological report text.

    Args:
        text_dir: Directory containing extracted .txt files
        output_dir: Where to save model and topic outputs
        num_topics: Number of latent topics to discover
        passes: Training passes (higher = better but slower)
        random_state: For reproducibility

    Returns:
        Dict with topic words and coherence score
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stopwords = build_geological_stopwords()
    dictionary, corpus, doc_labels, all_docs = build_corpus(text_dir, stopwords)

    log.info(f"Training LDA with {num_topics} topics, {passes} passes...")

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
        alpha="auto",
        eta="auto",
        minimum_probability=0.01,
    )

    # Compute coherence score
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=all_docs,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence_score = coherence_model.get_coherence()
    log.info(f"Coherence score (c_v): {coherence_score:.4f}")

    # Save model
    model_path = output_dir / "lda_model"
    lda_model.save(str(model_path))
    dictionary.save(str(output_dir / "dictionary.dict"))
    log.info(f"Model saved to {output_dir}")

    # Extract and display topics
    topics = {}
    log.info("\n=== DISCOVERED TOPICS ===")
    for topic_id in range(num_topics):
        top_words = lda_model.show_topic(topic_id, topn=10)
        words = [w for w, _ in top_words]
        topics[topic_id] = {
            "words": words,
            "label": _auto_label_topic(words),
        }
        log.info(f"Topic {topic_id:2d} [{topics[topic_id]['label']}]: {', '.join(words[:8])}")

    # Save topic summary
    summary = {
        "num_topics": num_topics,
        "coherence_score": coherence_score,
        "corpus_size": len(corpus),
        "dictionary_size": len(dictionary),
        "topics": topics,
    }
    with open(output_dir / "topic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Build document-topic matrix for GBM features
    doc_topic_matrix = []
    for doc_bow in corpus:
        topic_dist = dict(lda_model.get_document_topics(doc_bow, minimum_probability=0.0))
        row = [topic_dist.get(i, 0.0) for i in range(num_topics)]
        doc_topic_matrix.append(row)

    import numpy as np
    matrix = np.array(doc_topic_matrix)
    np.save(output_dir / "doc_topic_matrix.npy", matrix)
    log.info(f"Document-topic matrix: {matrix.shape} saved")

    return summary


def _auto_label_topic(words: list[str]) -> str:
    """Heuristic topic labeling based on top words."""
    word_set = set(words)

    label_rules = [
        ({"drill", "drillhole", "collar", "azimuth", "dip", "core", "intercept"}, "Drilling"),
        ({"grade", "cutoff_grade", "ore_grade", "assay", "sample", "analysis"}, "Assaying"),
        ({"flotation", "concentrate", "recovery", "reagent", "circuit", "mill"}, "Metallurgy"),
        ({"reserve", "resource", "indicated", "inferred", "measured", "estimation"}, "Resource Estimation"),
        ({"stope", "pillar", "development", "decline", "shaft", "underground"}, "Mine Design"),
        ({"tailings", "dam", "pond", "water", "environmental", "rehabilitation"}, "Environment"),
        ({"geology", "lithology", "stratigraphy", "structure", "fault", "formation"}, "Geology"),
        ({"cost", "capital", "operating", "economic", "revenue", "production"}, "Economics"),
        ({"power", "energy", "electrical", "diesel", "solar", "generator"}, "Infrastructure"),
        ({"geotechnical", "stability", "support", "rock", "strength", "failure"}, "Geotechnics"),
        ({"chalcopyrite", "bornite", "chalcocite", "sulphide", "oxide", "mineralization"}, "Mineralogy"),
        ({"water", "pumping", "dewatering", "aquifer", "groundwater", "hydrology"}, "Hydrology"),
        ({"transport", "road", "haul", "logistics", "port", "shipping"}, "Logistics"),
        ({"employee", "community", "social", "health", "safety", "permit"}, "Social/Permitting"),
        ({"spectral", "alteration", "band", "sentinel", "imagery", "remote"}, "Remote Sensing"),
    ]

    for keywords, label in label_rules:
        if word_set & keywords:
            return label

    return "General"


if __name__ == "__main__":
    summary = train_lda(
        text_dir="data/processed/text",
        output_dir="models/topic",
        num_topics=15,
        passes=10,
    )
    print(f"\nCoherence: {summary['coherence_score']:.4f}")
    print(f"Topics discovered: {summary['num_topics']}")
