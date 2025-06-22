import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# Load the GROQ API Key
load_dotenv()
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="gemma2-9b-it")

# Define a simple state structure
class SimpleState(TypedDict):
    user_input: str
    messages: List[str]
    final_output: str

# === Node 1: input_router ===
def input_router(state: SimpleState):
    user_text = state["user_input"]
    return {
        "messages": [f"Respond to this: {user_text}"]
    }

# === Node 2: responder ===
def responder(state: SimpleState):
    if state["messages"]:
        prompt = state["messages"][-1]
        reply = llm.invoke([{"role": "user", "content": prompt}])
        return {
            "messages": state["messages"] + [reply.content]
        }
    return {}

# === Node 3: output ===
def output_node(state: SimpleState):
    return {
        "final_output": f"Response:\n{state['messages'][-1]}"
    }

# === Routing logic ===
def always_continue(_): return "continue"

# === Build the graph ===
builder = StateGraph(SimpleState)
builder.add_node("input_router", input_router)
builder.add_node("responder", responder)
builder.add_node("output", output_node)

builder.set_entry_point("input_router")
builder.add_edge("input_router", "responder")
builder.add_edge("responder", "output")
builder.add_edge("output", END)

# Compile
graph = builder.compile()

# === Run it ===
def run_simple_network(user_input: str):
    initial_state = {
        "user_input": user_input,
        "messages": [],
        "final_output": ""
    }
    print(graph.get_graph().draw_mermaid())
    display(
        Image(
            graph.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )
    result = graph.invoke(initial_state)
    print(result["final_output"])



if __name__ == "__main__":
    while True:
        user_input = input("Enter text (or 'quit'): ")
        if user_input.lower() in ["quit", "q"]:
            break
        run_simple_network(user_input)
