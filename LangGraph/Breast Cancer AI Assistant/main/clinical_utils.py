import pandas as pd
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api, model_name="gemma2-9b-it")

HISTORY_FILE = "historical_clinical_profiles.csv"

def match_clinical_history(input_profile):
    df = pd.read_csv(HISTORY_FILE)

    matches = df[
        (df["gender"] == input_profile["gender"]) &
        (df["family_history"].str.contains(input_profile["family_history"].split()[0], na=False)) &
        (df["symptoms"].str.contains(input_profile["symptoms"].split()[0], na=False)) &
        (df["prior_illnesses"].str.contains(input_profile["prior_illnesses"].split()[0], na=False))
    ]

    if matches.empty:
        return {"match_found": False, "data": None}
    
    return {"match_found": True, "data": matches.iloc[0].to_dict()}


def get_clinical_history_response(input_profile):
    result = match_clinical_history(input_profile)

    if not result["match_found"]:
        prompt = (
            "No close match was found in the historical dataset for this patient profile:\n\n"
            f"{input_profile}\n\n"
            "Please offer general diagnostic advice based on the information provided."
        )
    else:
        match = result["data"]
        prompt = (
            f"You are a clinical assistant analyzing a patient's case based on historical records.\n\n"
            f"🧑 Patient: {input_profile}\n"
            f"📚 Closest Match Found:\n"
            f"- Age: {match['age']}\n"
            f"- Family History: {match['family_history']}\n"
            f"- Symptoms: {match['symptoms']}\n"
            f"- Prior Illnesses: {match['prior_illnesses']}\n"
            f"- Treatments: {match['treatments']}\n"
            f"- Outcome: **{match['outcome']}**\n\n"
            "Please analyze the matched case and advise the possible diagnosis, risk level, and next steps for the patient."
        )

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content
