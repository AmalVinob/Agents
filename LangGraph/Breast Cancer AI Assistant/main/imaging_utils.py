import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from roi_agent import classify_roi_image

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api, model_name="gemma2-9b-it")

def get_imaging_agent_response(image_path):
    # Run ROI classification
    result = classify_roi_image(image_path)
    label = result['prediction']
    confidence = result['confidence']

    # Generate prompt for LLM
    prompt = (
        "You are a radiology AI assistant interpreting a mammogram ROI.\n\n"
        f"🖼️ ROI Image Path: {result['image']}\n"
        f"🔍 The classification model (MobileNetV2) predicts:\n"
        f"• Diagnosis: **{label}**\n"
        f"• Confidence: **{confidence:.2%}**\n\n"
        "Please explain what this diagnosis could imply for the patient and suggest any recommended next steps or clinical actions."
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content
