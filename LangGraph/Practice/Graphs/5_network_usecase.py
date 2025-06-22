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
def inventory_monitor(state):
    return {"message": ["hello, this is inventory monitor"]}

def demand_forcasting(state):
    return {"message": ["i am forcasting the demand"]}

def Eoq_calculator(state):
    return {"message": ["i am calculating the EOQ"]}

def Discount_pricing(state):
    return {"message": ["discount pricing"]}

def shortage_alert_agent(state):
    return {"message": ["i am sending a storage alert"]}

def supplier_production(state):
    return {"message": ["i am supplier production"]}




    
def route_from_inventory(state: State) -> Literal["demand", "eoq", "shortage"]:
    if "hello, this is inventory monitor" in state["message"]:
        return "eoq"
    
    elif "i am forcasting the demand" in state["message"]:
        return "demand"
    
    else:
        return "shortage"
    
def route_from_demand(state: State) -> Literal["eoq", "discount"]:
    if "i am forcasting the demand" in state["message"]:
        return "eoq"
    
    else:
        return "discount"
    
def route_from_eoq(state: State) -> Literal["shortage", "supplier"]:
    if "i am calculating the EOQ" in state["message"]:
        return "shortage"
    
    else:
        return "supplier"

def route_from_shortage(state: State) -> Literal["supplier", "demand"]:
    if "i am sending a storage alert" in state["message"]:
        return "demand"
    else:
        return "supplier"

def route_from_supplier(state: State) -> Literal["eoq","discount", END]:
    if "i am supplier production" in state["message"]:
        return END
    elif "discount pricing" in state["message"]:
        return "eoq"
    else:   
        return "supplier"

# Add nodes
graph_builder.add_node("inventory", inventory_monitor)
graph_builder.add_node("demand", demand_forcasting)
graph_builder.add_node("eoq", Eoq_calculator)
graph_builder.add_node("discount", Discount_pricing)
graph_builder.add_node("shortage", shortage_alert_agent)
graph_builder.add_node("supplier", supplier_production)



# Define edges
graph_builder.add_edge(START, "inventory")
graph_builder.add_conditional_edges("inventory",route_from_inventory, {"demand": "demand", "eoq": "eoq", "shortage": "shortage"})
graph_builder.add_conditional_edges("demand", route_from_demand, {"eoq": "eoq", "discount": "discount"})
graph_builder.add_conditional_edges("eoq", route_from_eoq, {"shortage": "shortage", "supplier": "supplier"})
graph_builder.add_conditional_edges("shortage", route_from_shortage, {"supplier": "supplier", "demand": "demand"})
graph_builder.add_edge("discount", "eoq")
graph_builder.add_conditional_edges("supplier", route_from_supplier, {END: END, "eoq": "eoq", "discount": "discount"})




compiled_graph = graph_builder.compile()


print(compiled_graph.get_graph().draw_mermaid())


result = compiled_graph.invoke({"message": []})
print("\nFinal result:", result)
