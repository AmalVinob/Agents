import os
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, TypedDict, Literal
import operator

class TestState(TypedDict):
    message: Annotated[List[str], operator.add]


graph_builder = StateGraph(TestState)

# Define node functions
def node1(state):
    return {"message": ["hello this is from node 1"]}

def node2(state):
    return {"message": ["hello this is from node 2"]}

def node3(state):
    return {"message": ["hello this is from node 3"]}

def node4(state):
    return {"message": ["hello this is from node 4"]}

def node5(state):
    return {"message": ["hello this is from node 5"]}

# Define the routing function for node 3
def route_from_node3(state: TestState) -> Literal["node4", "node5"]:
    if len(state["message"]) % 2 == 0:
        return "node4"
    else:
        return "node5"

# Add nodes to the graph
graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)
graph_builder.add_node("node4", node4)
graph_builder.add_node("node5", node5)

# Add edges to create the network pattern
graph_builder.add_edge(START, "node1")


graph_builder.add_edge("node1", "node2")


graph_builder.add_edge("node2", "node3")

# node3 -> node4 or node5 (conditional)
graph_builder.add_conditional_edges(
    "node3", route_from_node3, {"node4": "node4", "node5": "node5"}
)


graph_builder.add_edge("node4", "node2")


graph_builder.add_edge("node5", END)


compiled_graph = graph_builder.compile()

# Display the graph
print(compiled_graph.get_graph().draw_mermaid())

# Run the graph
result = compiled_graph.invoke({"message": []})
print("\nFinal result:", result)