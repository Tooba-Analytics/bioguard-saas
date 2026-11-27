from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field

class PatientProfile(BaseModel):
    age: int = Field(default=0)
    sex: str = Field(default="Unknown")
    primary_condition: str = Field(description="Primary condition or research focus")
    comorbidities: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    lab_markers: Dict[str, str] = Field(default_factory=dict)

class CritiqueDecision(BaseModel):
    is_sufficient: bool
    reason: str
    broadened_condition: str

class AgentState(TypedDict):
    run_id: str
    raw_notes: str
    patient: Dict[str, Any]
    normalized_drugs: List[Dict[str, Any]]
    fda_records: List[Dict[str, Any]]
    trials: List[Dict[str, Any]]
    critique: Dict[str, Any]
    retry_count: int
    grounding: Dict[str, Any]
    final_report: str
    signal_score: int
    confidence_label: str
    trace_logs: List[str]