import os
from langgraph.graph import StateGraph, START, END
import random, operator
from typing import Annotated, List, TypedDict, Literal
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod

class TestState(TypedDict):
    message: Annotated[List[str], operator.add]

graph_builder = StateGraph(TestState)

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

def route(state: TestState):
    if random.choice([True, False]):
        return "node5"
    return END

# def route_2(state: TestState):
#     if random.choice([True, False]):
#         return "node3"
#     return END

# def routing_function(state: TestState) -> Literal["node2","node2"]:
#     if state['message'] == True:
#         return "node2"
#     else:
#         return "node3"
    

graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)
graph_builder.add_node("node4", node4)
graph_builder.add_node("node5", node5)

graph_builder.add_edge(START, "node1")
graph_builder.add_edge("node1", "node2")
graph_builder.add_edge("node2", "node3")
graph_builder.add_edge("node3", "node4")
graph_builder.add_edge("node5", "node4")

graph_builder.add_conditional_edges("node4", route, {"node5": "node5", END: END})
# graph_builder.add_conditional_edges("node1", route_2, {True: "node2", False: "node3"})
# graph_builder.add_conditional_edges("node1", routing_function, {True: "node2", False: "node3"})

# Compile and display the graph
compiled_graph = graph_builder.compile()
# display(Image(compiled_graph.get_graph().draw_mermaid_png()))
print(compiled_graph.get_graph().draw_mermaid())

result = compiled_graph.invoke({"message": []})
print(result)