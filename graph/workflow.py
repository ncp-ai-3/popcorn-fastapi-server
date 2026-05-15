"""LangGraph 워크플로 조립."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.nodes_generate import generate_recommendation
from graph.nodes_retrieval import retrieve_popups, retrieve_recent_three
from graph.nodes_router import prepare_embedding, router_node
from graph.routing import route_after_router
from graph.state import AgentState

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("prepare_embedding", prepare_embedding)
workflow.add_node("retrieve_popups", retrieve_popups)
workflow.add_node("retrieve_recent_three", retrieve_recent_three)
workflow.add_node("generate", generate_recommendation)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    route_after_router,
    {
        "prepare_embedding": "prepare_embedding",
        "retrieve_recent_three": "retrieve_recent_three",
    },
)
workflow.add_edge("prepare_embedding", "retrieve_popups")
workflow.add_edge("retrieve_popups", "generate")
workflow.add_edge("retrieve_recent_three", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
