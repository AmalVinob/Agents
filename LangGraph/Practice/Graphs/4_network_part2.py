import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator

# Define the shared state
class State(TypedDict):
    message: Annotated[list[str], operator.add]

# Build the graph
graph_builder = StateGraph(State)

# Node definitions
def node1(state):
    return {"message": state["message"] + ["hello, this is node 1"]}

def node2(state):
    return {"message": state["message"] + ["hello, i want to go to node 4"]}

def node3(state):
    return {"message": state["message"] + ["hello, this is node 3"]}

def node4(state):
    return {"message": state["message"] + ["hello, this is node 4"]}

def node5(state):
    return {"message": state["message"] + ["hello, this is node 5"]}

# Conditional routing from node3
def route_from_node3(state: State) -> Literal["node4", "node5"]:
    if "hello, i want to go to node 4" in state["message"]:
        return "node4"
    else:
        return "node5"

# Conditional routing from node5
def route_from_node5(state: State) -> Literal["node2", END]:
    if "hello, this is node 5" in state["message"]:
        return END
    else:
        return "node2"

# Add nodes
graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)
graph_builder.add_node("node4", node4)
graph_builder.add_node("node5", node5)

# Define edges
graph_builder.add_edge(START, "node1")
graph_builder.add_edge("node1", "node2")
graph_builder.add_edge("node2", "node3")
graph_builder.add_conditional_edges("node3", route_from_node3, {
    "node4": "node4",
    "node5": "node5"
})
graph_builder.add_edge("node4", "node5")
graph_builder.add_conditional_edges("node5", route_from_node5, {
    "node2": "node2",
    END: END
})

# Compile and run the graph
compiled_graph = graph_builder.compile()

# Optional: Draw the graph
print(compiled_graph.get_graph().draw_mermaid())

# Invoke the graph
result = compiled_graph.invoke({"message": []})
print("\nFinal result:", result)
