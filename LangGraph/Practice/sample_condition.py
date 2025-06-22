import os
from dotenv import load_dotenv
import sys

# from src.utils import State, GraphInput, GraphOutput, GraphConfig, check_chapter
# from src.nodes import *
# from src.routers import *
# from langgraph.graph import END
# from src.utils import State

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langgraph.graph import END

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def should_go_to_brainstorming_writer(state: State):
    if state.get('instructor_documents', '') == '':
        return "human_feedback"
    else:
        return "brainstorming_writer"
    
def should_continue_with_critique(state: State):
    if state.get('is_plan_approved', None) is None: 
        return "brainstorming_critique"
    elif state['is_plan_approved'] == True:
        return "writer"
    else:
        return "brainstorming_critique"
    
def has_writer_ended_book(state: State):
    if state['current_chapter'] == len(state['chapters_summaries']):
        return END
    else:
        return "writer"

# Initialize the ChatGroq model
# You might need to adjust the model name based on Groq's available models
groq_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.7,
    max_tokens=2048
)

# Update your node functions to use the Groq model (assuming they need the LLM)
# For example, you might need to modify your import in src.nodes to use the groq_llm

# workflow = StateGraph(State, 
#                       input=GraphInput,
#                       config_schema=GraphConfig)

# workflow.add_node("instructor", get_clear_instructions)
# workflow.set_entry_point("instructor")
# workflow.add_node("human_feedback", read_human_feedback)
# workflow.add_node("brainstorming_writer", making_writer_brainstorming)
# workflow.add_node("brainstorming_critique", brainstorming_critique)
# workflow.add_node("writer", generate_content)
# workflow.add_conditional_edges(
#     "instructor",
#     should_go_to_brainstorming_writer
# )
# workflow.add_edge("human_feedback", "instructor")
# workflow.add_conditional_edges(
#     "brainstorming_writer",
#     should_continue_with_critique
# )

workflow.add_edge("brainstorming_critique", "brainstorming_writer")
workflow.add_conditional_edges(
    "writer",
    has_writer_ended_book
)

app = workflow.compile(
    interrupt_before=['human_feedback']
)

# If you need to pass the LLM to your nodes, you might need to update your node functions
# or modify how you're adding nodes to the graph. Without seeing those implementations,
# I've provided the basic structure.

# Example of how you might run the workflow
if __name__ == "__main__":
    # Initial state setup
    initial_state = {
        "instructor_documents": "Your writing prompt or instructions here",
        "chapters_summaries": ["Chapter 1 summary", "Chapter 2 summary", "Chapter 3 summary"],
        "current_chapter": 0,
        "is_plan_approved": None
    }
    
    # Run the workflow
    # This is just an example and may need to be adjusted based on your actual implementation
    result = app.invoke(initial_state)
    print("Workflow completed with result:", result)