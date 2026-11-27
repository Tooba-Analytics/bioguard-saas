from datetime import datetime
import os
import tempfile
import duckdb
import pandas as pd

_DB_PATH = os.path.join(tempfile.gettempdir(), "bioguard_saas_ledger.duckdb")

def persist_run(run_id: str, condition: str, score: int, label: str, trials_n: int, grounded: bool, latency_s: float):
    try:
        conn = duckdb.connect(_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_ledger (
                run_id VARCHAR, logged_at VARCHAR, condition VARCHAR,
                signal_score INT, confidence VARCHAR, trials_found INT,
                grounding_ok BOOLEAN, latency_s DOUBLE
            )
        """)
        conn.execute(
            "INSERT INTO audit_ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), condition, score, label, trials_n, grounded, round(latency_s, 2)),
        )
        conn.close()
    except Exception:
        pass

def fetch_ledger() -> pd.DataFrame:
    try:
        conn = duckdb.connect(_DB_PATH)
        df = conn.execute("SELECT * FROM audit_ledger ORDER BY logged_at DESC LIMIT 30").fetchdf()
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["run_id", "logged_at", "condition", "signal_score", "confidence", "trials_found", "grounding_ok", "latency_s"])