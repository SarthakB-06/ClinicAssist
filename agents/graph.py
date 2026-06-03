from langgraph.graph import StateGraph, END
from agents.state import DischargeSummaryState


from agents.nodes.extractor import extraction_node
from agents.nodes.reconciliation import reconciliation_node
from agents.nodes.self_eval import self_evaluation_node
from agents.nodes.attribution import attribution_node
from agents.nodes.doctor_val import doctor_evaluation_node


def evaluation_router(state: DischargeSummaryState) -> str:
    if state.is_summary_complete:
        return "attribution"
    return "self_evaluation"


def build_discharge_summary_graph():
    workflow = StateGraph(DischargeSummaryState)

    workflow.add_node("extraction", extraction_node)
    workflow.add_node("reconciliation", reconciliation_node)
    workflow.add_node("self_evaluation", self_evaluation_node)
    workflow.add_node("attribution", attribution_node)
    workflow.add_node("doctor_evaluation", doctor_evaluation_node)

    workflow.set_entry_point("extraction")
    workflow.add_edge("extraction", "reconciliation")
    workflow.add_edge("reconciliation", "self_evaluation")

    workflow.add_conditional_edges(
        "self_evaluation",
        evaluation_router,
        {
            "self_evaluation": "self_evaluation",
            "attribution": "attribution"
        }
    )
    workflow.add_edge("attribution", "doctor_evaluation")
    workflow.add_edge("doctor_evaluation", END)
    return workflow.compile()
