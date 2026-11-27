from langgraph.graph import StateGraph, END
from src.config import MAX_CRITIC_RETRIES
from src.schemas import AgentState
from src.agents import triage_agent, parallel_evidence_agent, critic_agent, synthesizer_and_verifier_agent

def _route_critic(state: AgentState) -> str:
    sufficient = state.get("critique", {}).get("is_sufficient", True)
    retries = state.get("retry_count", 0)
    if (not sufficient) and (retries <= MAX_CRITIC_RETRIES):
        return "parallel_evidence"
    return "synthesizer"

def create_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_agent)
    graph.add_node("parallel_evidence", parallel_evidence_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("synthesizer", synthesizer_and_verifier_agent)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "parallel_evidence")
    graph.add_edge("parallel_evidence", "critic")
    graph.add_conditional_edges("critic", _route_critic)
    graph.add_edge("synthesizer", END)
    return graph.compile()