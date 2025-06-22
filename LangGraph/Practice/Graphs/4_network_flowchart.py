import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal, List
import operator

class State(TypedDict):
    message: Annotated[list[str], operator.add]

graph_builder = StateGraph(State)

def node1(state):
    return {"message": ["hello, this is node 1"]}

def node2(state):
    return {"message": ["hello, i want to go to node 4"]}

def node3(state):
    return {"message": ["hello, this is node 3"]}

def node4(state):
    return {"message": ["hello, this is node 4"]}

def node5(state):
    return {"message": ["hello, this is node 5"]}


def route_from_node3(state: State)-> Literal["node4", "node5"]:
    if state["message"] == ["hello, i want to go to node 4"]:
        return "continue"
    else:
        return "to_node5"


def route_from_node5(state: State)-> Literal["node2", END]:
    if state["message"] == ["hello, this is node 5"]:
        return END
    else:
        return "node2"



graph_builder.add_node("node1", node1)
graph_builder.add_node("node2", node2)
graph_builder.add_node("node3", node3)
graph_builder.add_node("node4", node4)
graph_builder.add_node("node5", node5)


graph_builder.add_edge(START, "node1")
graph_builder.add_edge("node1", "node2")
graph_builder.add_edge("node2", "node3")

#What will happen is we will call `route_from_node3`, and then the output of that
# will be matched against the keys in this mapping.
# Based on which one it matches, that node will then be called.
graph_builder.add_conditional_edges("node3", route_from_node3, {"continue" : "node4", "to_node5" : "node5"})
#graph_builder.add_conditional_edges("node3", route_from_node3, {"node4" : "node4", "node5" : "node5"})

graph_builder.add_edge("node4", "node5")
graph_builder.add_conditional_edges("node5", route_from_node5, {"node2" : "node2", END:END})
# graph_builder.add_edge("node5", END)

compiled_graph = graph_builder.compile()

print(compiled_graph.get_graph().draw_mermaid())

result = compiled_graph.invoke({"message": []})
print("\nFinal result:", result)