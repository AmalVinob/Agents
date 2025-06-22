import os
from langgraph.graph import StateGraph, START, END
import random
from typing import Annotated, List, TypedDict, Literal
from langchain_core.runnables.graph import MermaidDrawMethod
import operator

class TestState(TypedDict):
    message: Annotated[List[str], operator.add]
    counter: int

# Initialize the graph
graph_builder = StateGraph(TestState)

# Define node functions
def node1(state):
    current_counter = state.get("counter", 0)
    return {"message": ["hello this is from node 1"], "counter": current_counter + 1}

def node2(state):
    current_counter = state.get("counter", 0)
    return {"message": ["hello this is from node 2"], "counter": current_counter + 1}

def node3(state):
    current_counter = state.get("counter", 0)
    return {"message": ["hello this is from node 3"], "counter": current_counter + 1}

def node4(state):
    current_counter = state.get("counter", 0)
    return {"message": ["hello this is from node 4"], "counter": current_counter + 1}

def node5(state):
    current_counter = state.get("counter", 0)
    return {"message": ["hello this is from node 5"], "counter": current_counter + 1}

# Define routing functions
def route_from_node1(state: TestState) -> Literal["node2", "node3"]:
    # Route to either node2 or node3 from node1
    if random.choice([True, False]):
        return "node2"
    else:
        return "node3"

def route_from_node2(state: TestState) -> Literal["node4", "node5"]:
    # Route to either node4 or node5 from node2
    if random.choice([True, False]):
        return "node4"
    else:
        return "node5"

def route_from_node3(state: TestState) -> Literal["node4", "node5", "END"]:
    # Route to node4, node5, or END from node3
    choices = ["node4", "node5", END]
    return random.choice(choices)

def route_from_node4(state: TestState) -> Literal["node5", "node1", "END"]:
    # Route to node5, loop back to node1, or END from node4
    # This creates a potential cycle in the graph
    choices = ["node5", "node1", END]
    return random.choice(choices)

def route_from_node5(state: TestState) -> Literal["END", "node1"]:
    # Either end or loop back to node1 from node5
    if state["counter"] > 10:  # Prevent infinite loops
        return END
    elif random.choice([True, False]):
        return END
    else:
        return "node1"

# Add nodes to the graph
graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)
graph_builder.add_node("node4", node4)
graph_builder.add_node("node5", node5)

# Add edges to create a network pattern
graph_builder.add_edge(START, "node1")

# Add conditional edges
graph_builder.add_conditional_edges(
    "node1", route_from_node1, {"node2": "node2", "node3": "node3"}
)
graph_builder.add_conditional_edges(
    "node2", route_from_node2, {"node4": "node4", "node5": "node5"}
)
graph_builder.add_conditional_edges(
    "node3", route_from_node3, {"node4": "node4", "node5": "node5", END: END}
)
graph_builder.add_conditional_edges(
    "node4", route_from_node4, {"node5": "node5", "node1": "node1", END: END}
)
graph_builder.add_conditional_edges(
    "node5", route_from_node5, {"node1": "node1", END: END}
)

# Compile the graph
compiled_graph = graph_builder.compile()

# Display the graph
print(compiled_graph.get_graph().draw_mermaid())

# Run the graph
result = compiled_graph.invoke({"message": [], "counter": 0})
print("\nFinal result:", result)