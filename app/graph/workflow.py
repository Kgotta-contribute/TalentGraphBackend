from langgraph.graph import StateGraph, END
from app.graph.state import TalentAgentState

# Node implementations would go here or be imported
def analyze_jd_node(state: TalentAgentState): return state
def parse_resume_node(state: TalentAgentState): return state
def verify_requirements_node(state: TalentAgentState): return state
def verify_github_node(state: TalentAgentState): return state
def rank_candidate_node(state: TalentAgentState): return state
def generate_report_node(state: TalentAgentState): return state

def build_workflow() -> StateGraph:
    graph = StateGraph(TalentAgentState)
    
    # Add nodes
    graph.add_node("analyze_jd", analyze_jd_node)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("verify_requirements", verify_requirements_node)
    graph.add_node("verify_github", verify_github_node)  # conditional
    graph.add_node("rank_candidate", rank_candidate_node)
    graph.add_node("generate_report", generate_report_node)
    
    # Entry → JD analysis
    graph.set_entry_point("analyze_jd")
    
    # JD → Parse Resume
    graph.add_edge("analyze_jd", "parse_resume")
    
    # Parse → Verify
    graph.add_edge("parse_resume", "verify_requirements")
    
    # Verify → GitHub check (conditional)
    graph.add_conditional_edges(
        "verify_requirements",
        lambda state: "verify_github" if state.get("current_candidate_github_url") else "rank_candidate",
        {"verify_github": "verify_github", "rank_candidate": "rank_candidate"}
    )
    
    # GitHub → Rank
    graph.add_edge("verify_github", "rank_candidate")
    
    # Rank → Report
    graph.add_edge("rank_candidate", "generate_report")
    graph.add_edge("generate_report", END)
    
    return graph.compile()
