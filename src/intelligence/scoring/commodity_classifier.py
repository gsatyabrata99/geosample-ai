"""
Rule-based commodity classifier for NI 43-101 technical reports.

Detects the primary commodity before GBM scoring.
Reports not focused on copper are flagged with a warning
but still scored — the score just reflects geological data density
rather than copper-specific viability.
"""

import re
from dataclasses import dataclass

COPPER_TERMS = [
    "copper", "cu ", "% cu", "tcu", "chalcopyrite", "bornite", "chalcocite",
    "malachite", "cuprite", "covellite", "cubanite", "copper sulphide",
    "copper oxide", "copper concentrate", "copper deposit", "copperbelt",
    "porphyry copper", "sediment-hosted copper", "iocg",
]

GOLD_TERMS = [
    "gold", "au ", "g/t au", "oz au", "ounces of gold", "gold equivalent",
    "orogenic gold", "epithermal gold", "gold deposit", "gold mineralization",
]

SILVER_TERMS = ["silver", "ag ", "g/t ag", "ounces of silver"]
ZINC_TERMS = ["zinc", "zn ", "% zn", "zinc deposit", "vms", "massive sulphide"]
NICKEL_TERMS = ["nickel", "ni ", "% ni", "nickel deposit", "laterite"]
LITHIUM_TERMS = ["lithium", "li2o", "lithium carbonate", "spodumene", "brine"]
IRON_TERMS = ["iron ore", "magnetite", "hematite", "fe ", "% fe"]
COAL_TERMS = ["coal", "thermal coal", "metallurgical coal", "coking coal"]


@dataclass
class CommodityResult:
    primary_commodity: str
    is_copper: bool
    confidence: float  # 0-1
    copper_mentions: int
    competing_mentions: int
    warning: str | None


def classify_commodity(text: str) -> CommodityResult:
    """
    Classify the primary commodity in a geological report.
    
    Returns CommodityResult with is_copper flag and confidence score.
    """
    text_lower = text.lower()
    
    # Count mentions
    copper_count = sum(text_lower.count(term) for term in COPPER_TERMS)
    gold_count = sum(text_lower.count(term) for term in GOLD_TERMS)
    silver_count = sum(text_lower.count(term) for term in SILVER_TERMS)
    zinc_count = sum(text_lower.count(term) for term in ZINC_TERMS)
    nickel_count = sum(text_lower.count(term) for term in NICKEL_TERMS)
    lithium_count = sum(text_lower.count(term) for term in LITHIUM_TERMS)
    iron_count = sum(text_lower.count(term) for term in IRON_TERMS)
    coal_count = sum(text_lower.count(term) for term in COAL_TERMS)

    commodity_counts = {
        "Copper": copper_count,
        "Gold": gold_count,
        "Silver": silver_count,
        "Zinc": zinc_count,
        "Nickel": nickel_count,
        "Lithium": lithium_count,
        "Iron": iron_count,
        "Coal": coal_count,
    }

    total = sum(commodity_counts.values())
    if total == 0:
        return CommodityResult(
            primary_commodity="Unknown",
            is_copper=False,
            confidence=0.0,
            copper_mentions=0,
            competing_mentions=0,
            warning="No commodity terms detected. Report may not be a geological technical report.",
        )

    primary = max(commodity_counts, key=commodity_counts.get)
    primary_count = commodity_counts[primary]
    confidence = primary_count / total

    is_copper = primary == "Copper" and confidence > 0.3
    
    # Special case: copper-gold porphyry — still copper-relevant
    if primary == "Gold" and copper_count > 0:
        gold_copper_ratio = gold_count / max(copper_count, 1)
        if gold_copper_ratio < 3:
            is_copper = True
            primary = "Copper-Gold"

    competing = total - primary_count
    warning = None
    if not is_copper:
        warning = (
            f"This report appears to focus on {primary} ({primary_count} mentions) "
            f"rather than copper ({copper_count} mentions). "
            f"Viability score reflects geological data density, not copper-specific potential."
        )

    return CommodityResult(
        primary_commodity=primary,
        is_copper=is_copper,
        confidence=round(confidence, 3),
        copper_mentions=copper_count,
        competing_mentions=competing,
        warning=warning,
    )


if __name__ == "__main__":
    # Test on sample texts
    tests = [
        ("Copper test", "The Kamoa-Kakula deposit contains chalcopyrite and bornite. Copper grades of 2.86% Cu. Indicated resource of 1.8 billion tonnes at 2.86% TCu."),
        ("Gold test", "The Frotet project contains orogenic gold mineralization. Grades of 2.5 g/t Au. Resource of 2.55 million ounces of gold."),
        ("Zinc test", "The Estrades deposit is a VMS massive sulphide system with zinc, gold and silver. Zinc grades of 8% Zn."),
    ]
    
    for name, text in tests:
        result = classify_commodity(text)
        print(f"\n{name}:")
        print(f"  Primary: {result.primary_commodity}")
        print(f"  Is copper: {result.is_copper}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Copper mentions: {result.copper_mentions}")
        if result.warning:
            print(f"  Warning: {result.warning[:80]}...")
