"""
core/data_integrity.py
Scientific Data & Numerical Integrity Audit Engine for Academic Manuscripts.

Guarantees:
- Zero silent mutation of p-values, binding energies, concentrations, or percentages.
- Exact figure and table callout preservation (Figure X.X, Table X.X, Panel A-D).
- Protein active-site residue and PDB code fidelity (e.g. Cys166, Lys310, 2WYA).
- Quantitative fidelity scoring (0-100%) and actionable discrepancy reporting.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


# ==============================================================================
# Regex Patterns for Scientific & Empirical Data Extraction
# ==============================================================================

# Signed floats and scientific notation (e.g., -8.6, -9.0, +2.4, 1.2e-5, −8.6)
SIGNED_FLOAT_RE = re.compile(
    r"(?<![a-zA-Z0-9_])([−\-+]?\d+(?:\.\d+)?(?:[eE][\-+]?\d+)?)(?![a-zA-Z0-9_])"
)

# Percentages (e.g., 94.36%, 31.4%, 4.29 %)
PERCENTAGE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?\s*%)\b"
)

# P-values and statistical significance (e.g., p < 0.05, p = 0.001, p <= 0.01, P-value < 0.05)
P_VALUE_RE = re.compile(
    r"\b([pP](?:[\-\s]?value)?\s*(?:[<=><≥≤]|(?:\b(?:less|greater)\s+than\b))\s*\d+(?:\.\d+)?)\b"
)

# Common scientific units attached to numbers (e.g., -8.6 kJ/mol, 2.85 Å, 12.4 µM, 50 mg/kg)
UNIT_METRIC_RE = re.compile(
    r"([−\-+]?\d+(?:\.\d+)?)\s*(kJ/mol|kcal/mol|Å|angstroms?|µM|uM|nM|mM|mg/kg|mg/dL|kDa|Da|mol/L|h|hr|hrs|min|sec|%)\b",
    re.IGNORECASE
)

# Figure and Table Callouts (e.g., Figure 3.1, Fig. 2.1, Table 1.1, Panel A, 3D binding pose)
FIGURE_CALLOUT_RE = re.compile(
    r"\b((?:Figure|Fig\.)\s*\d+(?:\.\d+)?[a-zA-Z]?(?:\s*\([A-D]\))?)\b",
    re.IGNORECASE
)

TABLE_CALLOUT_RE = re.compile(
    r"\b(Table\s*\d+(?:\.\d+)?)\b",
    re.IGNORECASE
)

# Biological active-site residues and PDB codes (e.g., Cys166, Lys310, Phe304, Asp763, PDB: 2WYA)
RESIDUE_CODE_RE = re.compile(
    r"\b([A-Z][a-z]{2}\d{1,4}|PDB\s*[:\s]\s*[0-9][A-Za-z0-9]{3})\b"
)

# Publication years in citations (e.g. 1999, 2010, 2024)
CITATION_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Numbered citation bracket references (e.g. [1], [15], [1, 2], [3-5])
NUMBERED_CITATION_RE = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")


def extract_scientific_entities(text: str) -> Dict[str, List[str]]:
    """
    Deterministically extracts all scientific numbers, statistical metrics,
    figure/table callouts, and biochemical identifiers from text.
    """
    if not text:
        return {
            "p_values": [],
            "percentages": [],
            "unit_metrics": [],
            "raw_numbers": [],
            "figure_callouts": [],
            "table_callouts": [],
            "residues_and_pdb": [],
        }

    # Normalize unicode minus (−) to standard ASCII hyphen (-) for uniform matching
    norm_text = text.replace("−", "-").replace("–", "-")

    # Extract specific categories
    p_values = [m.strip() for m in P_VALUE_RE.findall(norm_text)]
    percentages = [m.strip() for m in PERCENTAGE_RE.findall(norm_text)]
    
    # Extract unit metrics (e.g., "-8.6 kJ/mol")
    unit_metrics = []
    for num, unit in UNIT_METRIC_RE.findall(norm_text):
        unit_metrics.append(f"{num} {unit}".lower())

    # Raw numbers (excluding standalone chapter numbers like 1, 2, 3 unless decimal or signed)
    raw_nums = []
    for n in SIGNED_FLOAT_RE.findall(norm_text):
        clean_n = n.strip()
        # Keep decimals, signed numbers, or numbers >= 10
        if "." in clean_n or clean_n.startswith(("-", "+")) or (clean_n.isdigit() and int(clean_n) >= 10):
            raw_nums.append(clean_n)

    fig_callouts = [re.sub(r"\s+", " ", m.strip()).title() for m in FIGURE_CALLOUT_RE.findall(text)]
    tbl_callouts = [re.sub(r"\s+", " ", m.strip()).title() for m in TABLE_CALLOUT_RE.findall(text)]
    residues = [m.strip() for m in RESIDUE_CODE_RE.findall(text)]
    citation_years = [m.strip() for m in CITATION_YEAR_RE.findall(norm_text)]

    numbered_cites = []
    for match in NUMBERED_CITATION_RE.finditer(norm_text):
        content = match.group(1)
        d = re.search(r'\d+', content)
        if d:
            numbered_cites.append((int(d.group(0)), match.group(0)))

    return {
        "p_values": sorted(list(set(p_values))),
        "percentages": sorted(list(set(percentages))),
        "unit_metrics": sorted(list(set(unit_metrics))),
        "raw_numbers": sorted(list(set(raw_nums))),
        "figure_callouts": sorted(list(set(fig_callouts))),
        "table_callouts": sorted(list(set(tbl_callouts))),
        "residues_and_pdb": sorted(list(set(residues))),
        "citation_years": sorted(list(set(citation_years))),
        "numbered_cites": numbered_cites,
    }


def audit_scientific_fidelity(original_text: str, humanized_text: str) -> Dict[str, Any]:
    """
    Compares the original text and the humanized output to verify that:
    1. Figure and Table callouts remain 100% faithful.
    2. Numerical metrics (binding energies, p-values, percentages) are not dropped or mutated.
    3. Active-site amino acid residues and target identifiers are fully preserved.

    Returns a quantitative fidelity score (0-100%) and an itemized discrepancy report.
    """
    orig_entities = extract_scientific_entities(original_text)
    trans_entities = extract_scientific_entities(humanized_text)

    discrepancies: List[Dict[str, str]] = []
    total_checks = 0
    passed_checks = 0

    # 1. Check Figure Callouts
    for fig in orig_entities["figure_callouts"]:
        total_checks += 1
        # Normalize comparison (e.g. 'Fig. 3.1' vs 'Figure 3.1')
        fig_num = re.search(r"\d+(?:\.\d+)?", fig)
        num_str = fig_num.group(0) if fig_num else ""
        
        found = any(num_str in t_fig for t_fig in trans_entities["figure_callouts"]) if num_str else False
        if found:
            passed_checks += 1
        else:
            discrepancies.append({
                "type": "Figure Callout",
                "original": fig,
                "status": "Missing or Altered Callout",
                "severity": "HIGH",
                "detail": f"Reference to '{fig}' was altered or omitted in enhanced text."
            })

    # 2. Check Table Callouts
    for tbl in orig_entities["table_callouts"]:
        total_checks += 1
        tbl_num = re.search(r"\d+(?:\.\d+)?", tbl)
        num_str = tbl_num.group(0) if tbl_num else ""
        found = any(num_str in t_tbl for t_tbl in trans_entities["table_callouts"]) if num_str else False
        if found:
            passed_checks += 1
        else:
            discrepancies.append({
                "type": "Table Callout",
                "original": tbl,
                "status": "Missing Table Reference",
                "severity": "HIGH",
                "detail": f"Reference to '{tbl}' was omitted in enhanced text."
            })

    # 3. Check Statistical P-values
    for pv in orig_entities["p_values"]:
        total_checks += 1
        # Extract numerical component of p-value
        pv_num = re.search(r"\d+(?:\.\d+)?", pv)
        num_str = pv_num.group(0) if pv_num else ""
        found = any(num_str in t_pv for t_pv in trans_entities["p_values"]) if num_str else False
        if found:
            passed_checks += 1
        else:
            discrepancies.append({
                "type": "Statistical P-Value",
                "original": pv,
                "status": "Missing Significance Value",
                "severity": "CRITICAL",
                "detail": f"P-value '{pv}' was mutated or omitted."
            })

    # 4. Check Unit Metrics (e.g. -8.6 kJ/mol)
    for um in orig_entities["unit_metrics"]:
        total_checks += 1
        # Check if identical or close unit metric exists
        num_part = um.split()[0]
        found = any(num_part in t_um for t_um in trans_entities["unit_metrics"])
        if not found:
            # Fallback check in raw text
            found = num_part in humanized_text.replace("−", "-")
        
        if found:
            passed_checks += 1
        else:
            discrepancies.append({
                "type": "Scientific Metric",
                "original": um,
                "status": "Missing Metric or Unit",
                "severity": "HIGH",
                "detail": f"Measured metric '{um}' was altered or omitted."
            })

    # 5. Check Active-Site Residues & PDB codes
    for res in orig_entities["residues_and_pdb"]:
        total_checks += 1
        if res.lower() in humanized_text.lower():
            passed_checks += 1
        else:
            discrepancies.append({
                "type": "Residue / Structure Code",
                "original": res,
                "status": "Missing Biochemical Identifier",
                "severity": "MEDIUM",
                "detail": f"Identifier '{res}' was omitted or renamed."
            })

    # 6. Check Raw Critical Numbers (e.g. 94.36, 145.91)
    for rn in orig_entities["raw_numbers"]:
        total_checks += 1
        if rn in humanized_text.replace("−", "-"):
            passed_checks += 1
        else:
            discrepancies.append({
                "type": "Numerical Data",
                "original": rn,
                "status": "Number Omitted or Rounded",
                "severity": "MEDIUM",
                "detail": f"Exact value '{rn}' was altered during rewriting."
            })

    # 7. Check Citation Years (Zero Year Drift)
    for yr in orig_entities["citation_years"]:
        total_checks += 1
        if yr in trans_entities["citation_years"] or yr in humanized_text:
            passed_checks += 1
        else:
            discrepancies.append({
                "type": "Citation Year",
                "original": yr,
                "status": "Publication Year Omitted",
                "severity": "CRITICAL",
                "detail": f"Publication year '{yr}' was omitted or altered during rewriting."
            })

    # 8. Check Sequential Ordering of First-Introduced Numbered Citations (IEEE / Vancouver)
    def _get_first_appearance_sequence(numbered_cites_list):
        seen = []
        for num, _ in numbered_cites_list:
            if num not in seen:
                seen.append(num)
        return seen

    orig_first_cites = _get_first_appearance_sequence(orig_entities.get("numbered_cites", []))
    trans_first_cites = _get_first_appearance_sequence(trans_entities.get("numbered_cites", []))

    if len(trans_first_cites) >= 2:
        for idx in range(len(trans_first_cites) - 1):
            curr_c = trans_first_cites[idx]
            next_c = trans_first_cites[idx + 1]
            if curr_c > next_c:
                # Check if this inversion was already present in the original document
                orig_had_inversion = False
                if curr_c in orig_first_cites and next_c in orig_first_cites:
                    orig_curr_idx = orig_first_cites.index(curr_c)
                    orig_next_idx = orig_first_cites.index(next_c)
                    if orig_curr_idx < orig_next_idx:
                        orig_had_inversion = True

                if not orig_had_inversion:
                    total_checks += 1
                    discrepancies.append({
                        "type": "Citation Sequence Inversion",
                        "original": f"[{curr_c}] before [{next_c}]",
                        "status": "Out-of-Order Numbered Citation",
                        "severity": "HIGH",
                        "detail": f"Numbered citation [{curr_c}] was newly introduced before [{next_c}], violating sequential academic ordering."
                    })

    # Calculate overall fidelity score
    if total_checks == 0:
        fidelity_score = 100.0
    else:
        fidelity_score = round((passed_checks / total_checks) * 100, 1)

    return {
        "fidelity_score": fidelity_score,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": len(discrepancies),
        "discrepancies": discrepancies,
        "original_summary": {
            "figures_count": len(orig_entities["figure_callouts"]),
            "tables_count": len(orig_entities["table_callouts"]),
            "p_values_count": len(orig_entities["p_values"]),
            "metrics_count": len(orig_entities["unit_metrics"]),
            "numbers_count": len(orig_entities["raw_numbers"]),
        },
        "is_safe_for_academic_submission": fidelity_score >= 95.0 and not any(d["severity"] == "CRITICAL" for d in discrepancies)
    }
