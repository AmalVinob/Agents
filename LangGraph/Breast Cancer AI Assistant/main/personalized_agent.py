import os
from dotenv import load_dotenv
import streamlit as st
import tempfile
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict
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
    message: Annotated[List[Dict], add_messages]  # Fixed annotation
    genomic_summary: str
    clinical_summary: str
    imaging_summary: str
    final_summary: str
    clinical_profile: dict
    variant_file_path: str
    image_file_path: str
 
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
 
# Define the agents as functions
def genomic_agent(state: AgentState) -> Dict:
    """Analyze genomic variants from uploaded file"""
    if not variant_file:
        genomic_summary = "No genomic file provided."
        return {
            "message": [{"role": "assistant", "content": f"[Genomic Agent]: {genomic_summary}"}],
            "genomic_summary": genomic_summary
        }
    
    try:
        g1000, clinvar = load_reference_data()
        patient_file = load_patient_file(variant_file, variant_file.name)
        report, summary = analyze_patient_variants(patient_file, g1000, clinvar)
        return {
            "message": [{"role": "assistant", "content": f"[Genomic Agent]: {report}"}],
            "genomic_summary": report
        }
    except Exception as e:
        error_msg = f"Error in genomic analysis: {str(e)}"
        return {
            "message": [{"role": "assistant", "content": f"[Genomic Agent]: {error_msg}"}],
            "genomic_summary": error_msg
        }
 
def clinical_agent(state: AgentState) -> Dict:
    """Analyze clinical history and provide recommendations"""
    try:
        profile = state["clinical_profile"]
        response = get_clinical_history_response(profile)
        return {
            "message": [{"role": "assistant", "content": f"[Clinical Agent]: {response}"}],
            "clinical_summary": response
        }
    except Exception as e:
        error_msg = f"Error in clinical analysis: {str(e)}"
        return {
            "message": [{"role": "assistant", "content": f"[Clinical Agent]: {error_msg}"}],
            "clinical_summary": error_msg
        }
 
def imaging_agent(state: AgentState) -> Dict:
    """Analyze medical imaging and provide insights"""
    if not image_file:
        imaging_summary = "No image file provided."
        return {
            "message": [{"role": "assistant", "content": f"[Imaging Agent]: {imaging_summary}"}],
            "imaging_summary": imaging_summary
        }
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_file.read())
            temp_image_path = tmp.name
        
        response = get_imaging_agent_response(temp_image_path)
        os.unlink(temp_image_path)  # Clean up temp file
        return {
            "message": [{"role": "assistant", "content": f"[Imaging Agent]: {response}"}],
            "imaging_summary": response
        }
    except Exception as e:
        error_msg = f"Error in imaging analysis: {str(e)}"
        return {
            "message": [{"role": "assistant", "content": f"[Imaging Agent]: {error_msg}"}],
            "imaging_summary": error_msg
        }
 
def final_summary_agent(state: AgentState) -> Dict:
    """Consolidate findings from all agents"""
    genomic_output = state['genomic_summary']
    clinical_output = state['clinical_summary']
    imaging_output = state['imaging_summary']
    clinical_profile = state["clinical_profile"]
    
    prompt = f"""
    You are an expert AI medical assistant. Below are findings from three diagnostic agents analyzing a patient's risk for breast cancer.
    Summarize the diagnostic findings and suggest next steps.
 
    🧬 Genomic Summary: 
    {genomic_output}
 
    🧑‍⚕️ Clinical Profile:
    Age: {clinical_profile['age']}
    Gender: {clinical_profile['gender']}
    Family History: {clinical_profile['family_history']}
    Symptoms: {clinical_profile['symptoms']}
    Prior Illnesses: {clinical_profile['prior_illnesses']}
    Treatments: {clinical_profile['treatments']}
 
    🧑‍⚕️ Clinical Agent Reply: 
    {clinical_output}
 
    🩻 Imaging Agent Reply: 
    {imaging_output}
 
    Provide a professional medical-style consolidated summary:
    - Diagnosis implications
    - Recommended referrals/treatments
    - Trial eligibility or genetic counseling
    """
    
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        final_summary = response.content
        return {
            "message": [{"role": "assistant", "content": f"[Final Summary]: {final_summary}"}],
            "final_summary": final_summary
        }
    except Exception as e:
        error_msg = f"Error generating final summary: {str(e)}"
        return {
            "message": [{"role": "assistant", "content": f"[Final Summary]: {error_msg}"}],
            "final_summary": error_msg
        }
 
# Run everything if submitted
if submitted:
    if not variant_file:
        st.warning("Please upload a genomic variant file.")
    elif not image_file:
        st.warning("Please upload a mammogram image.")
    else:
        with st.spinner("Running all agents in parallel..."):
            # Create initial state
            initial_state = {
                "message": [],  # Empty list for messages
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
                },
                "variant_file_path": "",
                "image_file_path": ""
            }
            
            # Create network-pattern graph
            graph = StateGraph(AgentState)
            
            # Add nodes
            graph.add_node("genomic_agent", genomic_agent)
            graph.add_node("clinical_agent", clinical_agent)
            graph.add_node("imaging_agent", imaging_agent)
            graph.add_node("final_summary_agent", final_summary_agent)
            
            # Configure parallel execution pattern
            graph.add_edge(START, "genomic_agent")
            graph.add_edge(START, "clinical_agent")
            graph.add_edge(START, "imaging_agent")
            
            # Connect all agents to final summary (network pattern)
            graph.add_edge("genomic_agent", "final_summary_agent")
            graph.add_edge("clinical_agent", "final_summary_agent")
            graph.add_edge("imaging_agent", "final_summary_agent")
            
            # Set exit point
            graph.add_edge("final_summary_agent", END)
            
            # Compile and run the graph
            workflow = graph.compile()
            final_state = workflow.invoke(initial_state)
            
            # Display results
            st.subheader("🧠 Consolidated AI Recommendation")
            st.success(final_state['final_summary'].replace("\n", " \n"))
            
            # Display individual agent outputs (collapsible)
            with st.expander("View Individual Agent Outputs"):
                st.subheader("🧬 Genomic Analysis")
                st.info(final_state['genomic_summary'])
                
                st.subheader("👩‍⚕️ Clinical Analysis")
                st.info(final_state['clinical_summary'])
                
                st.subheader("🩻 Imaging Analysis")
                st.info(final_state['imaging_summary'])
else:
    st.info("Please fill out the clinical form and upload all required files to begin analysis.")