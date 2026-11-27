import json
import time
import uuid
import streamlit as st
import plotly.graph_objects as go
from src.graph import create_workflow
from src.telemetry import persist_run, fetch_ledger

st.set_page_config(page_title="BioGuard AI", page_icon="🧬", layout="wide")

st.title("🧬 BioGuard AI — Autonomous Pharmacovigilance & Trial Intelligence")
st.warning(
    "**Research & Educational Disclaimer:** Built for exploratory biomedical data retrieval and workflow prototyping. "
    "Does **not** provide medical advice, diagnosis, treatment recommendations, or clinical trial eligibility determinations."
)

CASES = {
    "Case 1: Polypharmacy & Chronic Condition": (
        "64-year-old female with Type 2 Diabetes and Chronic Kidney Disease. "
        "Medications: Warfarin, Metformin, Lisinopril. Labs: HbA1c 8.7%, eGFR 44."
    ),
    "Case 2: Oncology Targeted Therapy": (
        "51-year-old male with Non-Small Cell Lung Cancer (EGFR exon 19 deletion), taking Osimertinib."
    ),
    "Case 3: Narrow Condition Self-Correction Demo": (
        "34-year-old female with Refractory Lupus Nephritis taking Mycophenolate Mofetil and Hydroxychloroquine."
    ),
}

selected = st.selectbox("Select Test Case:", list(CASES.keys()))
notes_input = st.text_area("Clinical / Research Notes:", value=CASES[selected], height=100)

if "run_history" not in st.session_state:
    st.session_state.run_history = []

if st.button("🚀 Run Autonomous Multi-Agent Audit", type="primary"):
    run_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    app_graph = create_workflow()

    with st.status(f"Executing Run `{run_id}`...", expanded=True) as status:
        state_out = app_graph.invoke({"run_id": run_id, "raw_notes": notes_input, "trace_logs": []})
        for line in state_out.get("trace_logs", []):
            st.write(line)
        status.update(label=f"Run `{run_id}` Complete", state="complete", expanded=False)

    latency = time.time() - t0
    patient_info = state_out.get("patient", {})
    cond_name = patient_info.get("primary_condition", "Unknown")
    grounded_flag = bool(state_out.get("grounding", {}).get("passed", True))

    persist_run(run_id, cond_name, state_out["signal_score"], state_out["confidence_label"], len(state_out.get("trials", [])), grounded_flag, latency)
    st.session_state.run_history.insert(0, {
        "run_id": run_id, "condition": cond_name, "signal_score": state_out["signal_score"],
        "confidence": state_out["confidence_label"], "grounded": grounded_flag, "latency_s": round(latency, 2),
    })

    tab1, tab2, tab3 = st.tabs(["📋 Grounded Report & Exports", "🔬 Raw & Normalized API Evidence", "📊 Production Ledger"])

    with tab1:
        c_left, c_right = st.columns([1, 2.5])
        with c_left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=state_out["signal_score"],
                title={"text": "Exploratory Signal Index"}, gauge={"axis": {"range": [0, 100]}},
            ))
            fig.update_layout(height=240, margin=dict(l=15, r=15, t=35, b=15))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Coverage: **{state_out['confidence_label']}**")
            st.caption(f"Citation Grounding: **{'PASSED' if grounded_flag else 'FLAGGED'}**")
            st.metric("End-to-End Latency", f"{latency:.2f}s")
            st.json(patient_info)

        with c_right:
            st.markdown(state_out["final_report"])
            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download Report (.md)", data=state_out["final_report"].encode("utf-8"), file_name=f"bioguard_{run_id}.md", mime="text/markdown")
            with d2:
                st.download_button("Download Audit Payload (.json)", data=json.dumps(state_out, indent=2, default=str).encode("utf-8"), file_name=f"bioguard_{run_id}.json", mime="application/json")

    with tab2:
        st.subheader("NIH RxNorm Drug Normalization")
        st.json(state_out.get("normalized_drugs", []))
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("OpenFDA Sampled Events")
            st.json(state_out.get("fda_records", []))
        with col2:
            st.subheader("ClinicalTrials.gov Recruiting Studies")
            st.json(state_out.get("trials", []))

    with tab3:
        st.subheader("Session Runs")
        st.dataframe(st.session_state.run_history, use_container_width=True)
        st.subheader("DuckDB Audit Ledger")
        ledger_df = fetch_ledger()
        st.dataframe(ledger_df, use_container_width=True)
        if not ledger_df.empty:
            st.download_button("Export Ledger (.csv)", data=ledger_df.to_csv(index=False).encode("utf-8"), file_name="bioguard_ledger.csv", mime="text/csv")