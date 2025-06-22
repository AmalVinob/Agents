import os
from dotenv import load_dotenv
import streamlit as st
import tempfile
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from clinical_utils import get_clinical_history_response
from genome_utils import load_reference_data, load_patient_file, analyze_patient_variants
from imaging_utils import get_imaging_agent_response
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api, model_name="gemma2-9b-it")

# Define the shared agent state
class AgentState(TypedDict):
    message: Annotated[list, add_messages]
    genomic_summary: str
    clinical_summary: str
    imaging_summary: str
    final_summary: str
    clinical_profile: dict

# UI setup
st.set_page_config(page_title="Medical Agent System", layout="wide")
st.title("🧬 Medical AI Assistant – Genomic, Clinical & Imaging AI Agents")
st.markdown("Upload patient files below. All agents will run in parallel and generate a consolidated recommendation.")

# Layout for file inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Genomic Variant File")
    variant_file = st.file_uploader("Upload VCF or CSV", type=["csv", "vcf"])

with col2:
    st.subheader("2️⃣ Mammogram ROI Image")
    image_file = st.file_uploader("Upload JPEG/PNG Image", type=["jpg", "jpeg", "png"])

# Clinical profile input
st.subheader("3️⃣ Clinical Profile")
with st.form("clinical_form"):
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_history = st.text_input("Family History (e.g., Mother breast cancer)")
    symptoms = st.text_input("Symptoms (e.g., Fatigue)")
    prior_illnesses = st.text_input("Prior Illnesses (e.g., Hypertension)")
    treatments = st.text_input("Treatments (e.g., None)")
    submitted = st.form_submit_button("🚀 Run All Agents")

# Initialize StateGraph
graph = StateGraph(state_class=AgentState, start_state=START)

# Define the agents as functions
def genomic_agent(state: AgentState):
    # Check if files are provided and valid
    if not variant_file:
        state['genomic_summary'] = "No genomic file provided."
        return state
    g1000, clinvar = load_reference_data()  # Your logic here
    patient_file = load_patient_file(variant_file, variant_file.name)  # Replace with actual path
    report, summary = analyze_patient_variants(patient_file, g1000, clinvar)
    state['genomic_summary'] = report
    return state

def clinical_agent(state: AgentState):
    profile = state["clinical_profile"]
    response = get_clinical_history_response(profile)  # Your logic here
    state['clinical_summary'] = response
    return state

def imaging_agent(state: AgentState):
    if not image_file:
        state['imaging_summary'] = "No image file provided."
        return state
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_file.read())
        temp_image_path = tmp.name
    response = get_imaging_agent_response(temp_image_path)  # Your logic here
    state['imaging_summary'] = response
    return state

# Final summary agent using the Groq API
def get_summary_agent_response(genomic_output, clinical_output, imaging_output, clinical_profile):
    prompt = f"""
You are an expert AI medical assistant. Below are findings from three diagnostic agents analyzing a patient's risk for breast cancer. Summarize the diagnostic findings and suggest next steps.

🧬 Genomic Summary:
{genomic_output[0]}

🧑‍⚕️ Clinical Profile:
{clinical_profile}

🧑‍⚕️ Clinical Agent Reply:
{clinical_output[-1].content}

🩻 Imaging Agent Reply:
{imaging_output[-1].content}

Provide a professional medical-style consolidated summary:
- Diagnosis implications
- Recommended referrals/treatments
- Trial eligibility or genetic counseling
"""

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content

# Final summary agent function
def final_summary_agent(state: AgentState):
    genomic_output = state['genomic_summary']
    clinical_output = state['clinical_summary']
    imaging_output = state['imaging_summary']
    clinical_profile = state["clinical_profile"]

    final_summary = get_summary_agent_response(genomic_output, clinical_output, imaging_output, clinical_profile)
    state['final_summary'] = final_summary
    return state

# Connect the agents to the graph
graph.add_node(genomic_agent, start=START)
graph.add_node(clinical_agent)
graph.add_node(imaging_agent)
graph.add_node(final_summary_agent, end=END)

# Create edges to connect agents to the final summary agent
graph.add_edge("genomic_summary", "final_summary")
graph.add_edge("clinical_summary", "final_summary")
graph.add_edge("imaging_summary", "final_summary")
graph.add_edge("final_summary", END)

# Run everything
if submitted and variant_file and image_file:
    with st.spinner("Running all agents in parallel..."):
        initial_state = {
            "message": [],
            "genomic_summary": "",
            "clinical_summary": "",
            "imaging_summary": "",
            "final_summary": "",
            "clinical_profile": {
                "age": age,
                "gender": gender,
                "family_history": family_history,
                "symptoms": symptoms,
                "prior_illnesses": prior_illnesses,
                "treatments": treatments
            }
        }
        # Run the graph and get the final summary
        state = graph.run(initial_state)
        st.subheader("🧠 Consolidated AI Recommendation")
        st.success(state['final_summary'].replace("\n", "  \n"))

else:
    st.warning("Please upload all required inputs to begin.")
