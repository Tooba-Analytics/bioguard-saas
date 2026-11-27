from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.config import API_TIMEOUT_SEC, MAX_RETRIES, MAX_WORKERS

def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=MAX_RETRIES, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def normalize_drug_rxnorm(drug_name: str) -> dict:
    clean = drug_name.strip()
    if not clean:
        return {"raw_name": drug_name, "normalized_name": "", "rxcui": None, "status": "EMPTY"}
    url = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    try:
        r = _session().get(url, params={"term": clean, "maxEntries": 1}, timeout=API_TIMEOUT_SEC)
        if r.status_code == 200:
            candidates = r.json().get("approximateGroup", {}).get("candidate", [])
            if candidates:
                top = candidates[0]
                return {
                    "raw_name": clean,
                    "rxcui": top.get("rxcui"),
                    "normalized_name": top.get("name") or clean,
                    "status": "NORMALIZED",
                }
        return {"raw_name": clean, "rxcui": None, "normalized_name": clean, "status": "UNRESOLVED"}
    except Exception as exc:
        return {"raw_name": clean, "rxcui": None, "normalized_name": clean, "status": "ERROR", "error": str(exc)}

def fetch_fda_events(drug_query: str) -> dict:
    clean = drug_query.strip()
    if not clean:
        return {"drug": drug_query, "status": "INVALID", "sample_size": 0, "serious_count": 0, "reactions": []}
    url = "https://api.fda.gov/drug/event.json"
    params = {"search": f'patient.drug.medicinalproduct:"{clean}"', "limit": 5}
    try:
        r = _session().get(url, params=params, timeout=API_TIMEOUT_SEC)
        if r.status_code == 200:
            rows = r.json().get("results", [])
            rx_set = set()
            serious = 0
            for item in rows:
                if str(item.get("serious")) == "1":
                    serious += 1
                for rx in item.get("patient", {}).get("reaction", []):
                    if rx.get("reactionmeddrapt"):
                        rx_set.add(rx["reactionmeddrapt"])
            return {
                "drug": clean,
                "status": "VERIFIED",
                "sample_size": len(rows),
                "serious_count": serious,
                "reactions": sorted(list(rx_set))[:6],
            }
        return {"drug": clean, "status": "NO_MATCH", "sample_size": 0, "serious_count": 0, "reactions": []}
    except Exception as exc:
        return {"drug": clean, "status": "ERROR", "sample_size": 0, "serious_count": 0, "reactions": [], "error": str(exc)}

def fetch_trials(condition: str) -> list:
    clean = condition.strip()
    if not clean:
        return []
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.cond": clean, "filter.overallStatus": "RECRUITING", "pageSize": 4}
    try:
        r = _session().get(url, params=params, timeout=API_TIMEOUT_SEC)
        if r.status_code == 200:
            studies = r.json().get("studies", [])
            out = []
            for s in studies:
                proto = s.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                elig = proto.get("eligibilityModule", {})
                out.append({
                    "nct_id": ident.get("nctId", "N/A"),
                    "title": ident.get("briefTitle", "Untitled"),
                    "min_age": elig.get("minimumAge", "Not specified"),
                    "sex": elig.get("sex", "ALL"),
                })
            return out
        return []
    except Exception:
        return []

def collect_parallel_biomedical_evidence(medications: list, condition: str) -> tuple[list, list, list]:
    norm_results, fda_results = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        norm_futs = {pool.submit(normalize_drug_rxnorm, m): m for m in medications}
        trial_fut = pool.submit(fetch_trials, condition)

        for fut in as_completed(norm_futs):
            norm_results.append(fut.result())
        trials_results = trial_fut.result()

        fda_targets = [
            n["normalized_name"] if n.get("normalized_name") else n["raw_name"]
            for n in norm_results
        ]
        fda_futs = {pool.submit(fetch_fda_events, t): t for t in fda_targets}
        for fut in as_completed(fda_futs):
            fda_results.append(fut.result())

    raw_order = {m.strip().lower(): i for i, m in enumerate(medications)}
    norm_results.sort(key=lambda x: raw_order.get(x["raw_name"].lower(), 999))
    return norm_results, fda_results, trials_results