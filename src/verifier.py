import re

NCT_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)

def verify_report_grounding(report_text: str, retrieved_trials: list) -> dict:
    allowed_ids = {
        t.get("nct_id", "").upper()
        for t in retrieved_trials
        if t.get("nct_id") and t.get("nct_id") != "N/A"
    }
    mentioned_ids = {m.group(0).upper() for m in NCT_PATTERN.finditer(report_text or "")}
    verified = sorted(list(mentioned_ids.intersection(allowed_ids)))
    unverified = sorted(list(mentioned_ids.difference(allowed_ids)))

    return {
        "passed": len(unverified) == 0,
        "verified_nct_ids": verified,
        "unverified_nct_ids_flagged": unverified,
        "notes": (
            f"Flagged unverified NCT IDs: {unverified}"
            if unverified
            else "All NCT citations verified against live ClinicalTrials.gov records."
        ),
    }