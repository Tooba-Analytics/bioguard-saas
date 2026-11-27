from langchain_groq import ChatGroq
from src.config import DEFAULT_MODEL, MAX_CRITIC_RETRIES
from src.schemas import AgentState, PatientProfile, CritiqueDecision
from src.tools import collect_parallel_biomedical_evidence
from src.verifier import verify_report_grounding

llm = ChatGroq(model=DEFAULT_MODEL, temperature=0.1)

def _score_evidence(meds: list, fda: list, trials: list) -> tuple[int, str]:
    verified = [x for x in fda if x.get("status") == "VERIFIED"]
    samples = sum(x.get("sample_size", 0) for x in verified)
    serious = sum(x.get("serious_count", 0) for x in verified)

    med_pts = min(25, len(meds) * 6)
    serious_pts = int((serious / max(1, samples)) * 45)
    trial_pts = min(20, len(trials) * 5)
    score = max(5, min(95, 10 + med_pts + serious_pts + trial_pts))

    ratio = len(verified) / max(1, len(meds))
    if ratio >= 0.75 and len(trials) > 0:
        label = "High Evidence Coverage"
    elif ratio >= 0.35 or len(trials) > 0:
        label = "Partial Evidence Coverage"
    else:
        label = "Sparse Evidence (Manual Lookup Needed)"
    return score, label

def triage_agent(state: AgentState):
    extractor = llm.with_structured_output(PatientProfile)
    profile = extractor.invoke(f"Extract structured biomedical & research fields from:\n{state['raw_notes']}")
    logs = list(state.get("trace_logs", []))
    logs.append(f"[Triage Agent] Condition='{profile.primary_condition}' | Meds={profile.medications}")
    return {"patient": profile.model_dump(), "retry_count": 0, "trace_logs": logs}

def parallel_evidence_agent(state: AgentState):
    patient = state["patient"]
    retry = state.get("retry_count", 0)
    cond_query = (
        state["critique"]["broadened_condition"]
        if retry > 0 and state.get("critique", {}).get("broadened_condition")
        else patient["primary_condition"]
    )
    norm_drugs, fda_data, trials_data = collect_parallel_biomedical_evidence(patient.get("medications", []), cond_query)
    logs = list(state.get("trace_logs", []))
    logs.append(f"[Parallel Retrieval] Query='{cond_query}' | RxNorm={len(norm_drugs)} | OpenFDA={len(fda_data)} | Trials={len(trials_data)}")
    return {"normalized_drugs": norm_drugs, "fda_records": fda_data, "trials": trials_data, "trace_logs": logs}

def critic_agent(state: AgentState):
    critic = llm.with_structured_output(CritiqueDecision)
    prompt = f"""
    Audit retrieval completeness:
    Primary Condition: {state['patient']['primary_condition']}
    Trials Found: {len(state['trials'])}
    Retry Count: {state['retry_count']}

    If Trials Found == 0 and Retry Count < {MAX_CRITIC_RETRIES}, set is_sufficient=False and supply a broader medical term in broadened_condition.
    Otherwise set is_sufficient=True and set broadened_condition to Primary Condition.
    """
    decision = critic.invoke(prompt)
    logs = list(state.get("trace_logs", []))
    will_retry = (not decision.is_sufficient) and (state["retry_count"] < MAX_CRITIC_RETRIES)
    logs.append(f"[{'Critic -> Retry' if will_retry else 'Critic -> Proceed'}] {decision.reason}")
    return {"critique": decision.model_dump(), "retry_count": state["retry_count"] + 1, "trace_logs": logs}

def synthesizer_and_verifier_agent(state: AgentState):
    score, label = _score_evidence(state["patient"].get("medications", []), state["fda_records"], state["trials"])
    prompt = f"""
    You are a Biomedical Research Summarizer.
    Rely STRICTLY on the JSON below. Do NOT invent NCT IDs or medical advice.

    Patient Profile: {state['patient']}
    RxNorm Normalized Drugs: {state['normalized_drugs']}
    OpenFDA Sampled Records: {state['fda_records']}
    Recruiting Trials: {state['trials']}
    Evidence Coverage: {label} (Signal Index: {score}/100)

    Write a clean Markdown summary with these exact headings:
    ### 1. Research Overview & Evidence Limitations
    ### 2. Drug Normalization & Sampled OpenFDA Terms
    ### 3. Recruiting ClinicalTrials.gov Entries
    ### 4. Follow-Up Questions for Domain Experts
    """
    draft = llm.invoke(prompt).content
    grounding = verify_report_grounding(draft, state["trials"])
    final_text = draft if grounding["passed"] else f"{draft}\n\n> **Grounding Warning:** {grounding['notes']}"

    logs = list(state.get("trace_logs", []))
    logs.append(f"[Verifier & Synthesizer] Grounding={grounding['passed']} | Signal={score}/100 ({label}).")
    return {
        "grounding": grounding,
        "final_report": final_text,
        "signal_score": score,
        "confidence_label": label,
        "trace_logs": logs,
    }