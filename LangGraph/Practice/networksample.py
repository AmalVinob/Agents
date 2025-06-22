import os
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, TypedDict, Literal
import operator
import random

class TestState(TypedDict):
    message: Annotated[List[str], operator.add]
    inventory_level: int
    iteration_count: int  # Track iterations to prevent infinite loops

# Initialize the graph
graph_builder = StateGraph(TestState)

# Define node functions (simplified)
def node1(state):  # Inventory Monitoring
    count = state.get("iteration_count", 0) + 1
    return {
        "message": [f"Processing at Node 1 (Inventory Monitoring) - iteration {count}"],
        "iteration_count": count
    }

def node2(state):  # Demand Forecasting
    count = state.get("iteration_count", 0)
    return {
        "message": [f"Processing at Node 2 (Demand Forecasting) - iteration {count}"]
    }

def node3(state):  # EOQ Calculator
    count = state.get("iteration_count", 0)
    return {
        "message": [f"Processing at Node 3 (EOQ Calculator) - iteration {count}"]
    }

def node4(state):  # Discount Pricing
    count = state.get("iteration_count", 0)
    return {
        "message": [f"Processing at Node 4 (Discount Pricing) - iteration {count}"]
    }

def node5(state):  # Shortage Alert
    count = state.get("iteration_count", 0)
    return {
        "message": [f"Processing at Node 5 (Shortage Alert) - iteration {count}"]
    }

def node6(state):  # Supplier/Production
    count = state.get("iteration_count", 0)
    return {
        "message": [f"Processing at Node 6 (Supplier/Production) - iteration {count}"]
    }

def node7(state):  # External Coordination
    count = state.get("iteration_count", 0)
    return {
        "message": [f"Processing at Node 7 (External Coordination) - iteration {count}"]
    }

# Define routing functions with loop prevention
def route_from_node1(state: TestState) -> Literal["node2", "node3", "node5"]:
    # Simple distribution of traffic
    r = random.random()
    if r < 0.4:
        return "node2"
    elif r < 0.7:
        return "node3"
    else:
        return "node5"

def route_from_node3(state: TestState) -> Literal["node5", "node6"]:
    # Route based on random condition
    if random.random() < 0.6:
        return "node5"
    else:
        return "node6"

def route_from_node5(state: TestState) -> Literal["node6", "END"]:
    # Higher chance to end as iterations increase
    if state["iteration_count"] > 3 or random.random() < 0.7:
        return END
    else:
        return "node6"

def route_from_node6(state: TestState) -> Literal["node2", "node7"]:
    # End the loop after several iterations
    if state["iteration_count"] > 2 or random.random() < 0.6:
        return "node7"  # Go to node7 which will end the process
    else:
        return "node2"  # Continue the loop

# Add nodes to the graph
graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)
graph_builder.add_node("node4", node4)
graph_builder.add_node("node5", node5)
graph_builder.add_node("node6", node6)
graph_builder.add_node("node7", node7)

# Add edges to create the network pattern based on the diagram
# Start -> node1 (Inventory Monitoring)
graph_builder.add_edge(START, "node1")

# Conditional edges from node1
graph_builder.add_conditional_edges(
    "node1", route_from_node1, {"node2": "node2", "node3": "node3", "node5": "node5"}
)

# node2 -> node3, node2 -> node4
graph_builder.add_edge("node2", "node3")
graph_builder.add_edge("node2", "node4")

# node4 -> node6
graph_builder.add_edge("node4", "node6")

# Conditional edges from node3
graph_builder.add_conditional_edges(
    "node3", route_from_node3, {"node5": "node5", "node6": "node6"}
)

# Conditional edges from node5
graph_builder.add_conditional_edges(
    "node5", route_from_node5, {"node6": "node6", END: END}
)

# Conditional edges from node6 to prevent infinite loops
graph_builder.add_conditional_edges(
    "node6", route_from_node6, {"node2": "node2", "node7": "node7"}
)

# node7 -> END
graph_builder.add_edge("node7", END)

# Compile the graph with increased recursion limit
graph_config = {"recursion_limit": 50}
compiled_graph = graph_builder.compile(config=graph_config)

# Display the graph visualization code
print(compiled_graph.get_graph().draw_mermaid())

# Run the graph
try:
    result = compiled_graph.invoke({"message": [], "inventory_level": 100, "iteration_count": 0})
    print("\nFinal result:", result)
except Exception as e:
    print(f"\nError occurred: {e}")
    print("\nTo fix recursion issues, try increasing the recursion_limit further or modify routing logic.")