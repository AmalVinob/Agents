from langchain_core.tools import tool
import re
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")


@tool
def replace_sim(phone_model: str) -> str:
    """Provides instructions for replacing a SIM card in a specific phone model."""
    instructions = {
        "iPhone 13": "1. Power off your iPhone 13.\n2. Locate the SIM tray on the right side of the phone.\n3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray.\n4. Gently push until the tray pops out.\n5. Remove the old SIM card from the tray and place the new SIM card in the tray.\n6. Insert the SIM tray back into the phone.\n7. Power on your iPhone 13.",
        "Samsung Galaxy S21": "1. Power off your Samsung Galaxy S21.\n2. Locate the SIM tray on the top of the phone.\n3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray.\n4. Gently push until the tray pops out.\n5. Remove the old SIM card from the tray and place the new SIM card in the tray.\n6. Insert the SIM tray back into the phone.\n7. Power on your Samsung Galaxy S21.",
        "Google Pixel 7": "1. Power off your Google Pixel 7.\n2. Locate the SIM tray on the left side of the phone.\n3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray.\n4. Gently push until the tray pops out.\n5. Remove the old SIM card from the tray and place the new SIM card in the tray, making sure it's aligned correctly.\n6. Insert the SIM tray back into the phone.\n7. Power on your Google Pixel 7.",
        "OnePlus 10 Pro": "1. Power off your OnePlus 10 Pro.\n2. Locate the SIM tray on the left side of the phone.\n3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray.\n4. Gently push until the tray pops out.\n5. Remove the old SIM card from the tray and place the new SIM card in the tray.\n6. Insert the SIM tray back into the phone.\n7. Power on your OnePlus 10 Pro.",
        "Xiaomi 12": "1. Power off your Xiaomi 12.\n2. Locate the SIM tray on the left side of the phone.\n3. Use a SIM ejector tool (or a small paperclip) to insert into the small hole on the SIM tray.\n4. Gently push until the tray pops out.\n5. Remove the old SIM card from the tray and place the new SIM card in the tray.\n6. Insert the SIM tray back into the phone.\n7. Power on your Xiaomi 12."
    }
    return instructions.get(phone_model, "Sorry, I don't have instructions for that phone model yet.")



def is_uncertain(response: str) -> bool:
    """
    Determines if an LLM response indicates uncertainty or inability to provide accurate information.
    """
    normalized_text = response.lower().strip()
    uncertainty_indicators = [
        "not sure", "don't know", "do not know", "doubt", "unable to confirm", "unable to determine",
        "might", "may", "could", "possibly", "somewhat",
        "limited access", "limited information", "outside my knowledge",
        "i think", "i believe", "i suppose", "i assume", "i guess", "i imagine", "i suspect",
        "i'm not sure", "i'm not certain", "i'm not confident",
        "as an ai", "being an ai", "as a language model", "being a language model",
        "this is beyond my scope", "this is outside my knowledge", "this seems too complex",
        "can you please be more specific", "can you provide more context",
        "i am a large language model", "for safety reasons", "for ethical reasons", "for security reasons"
    ]
    for indicator in uncertainty_indicators:
        if indicator in normalized_text:
            return True
    return False

llm = ChatGroq(model_name="llama3-8b-8192")

# Bind the replace_sim tool
llm_with_tools = llm.bind_tools([
    Tool.from_function(
        replace_sim,
        name="replace_sim",
        description="Provides step-by-step SIM card replacement instructions for a given phone model"
    )
])



def main():
    print("Welcome to the ChatBot! Type 'exit' to quit.")
    conversation_history = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Add user message to conversation history
        conversation_history.append(HumanMessage(content=user_input))

        # Invoke the LLM with tools
        response = llm_with_tools.invoke(conversation_history)

        # Check for uncertainty
        if is_uncertain(response.content):
            print("\nAI (uncertain):", response.content)
            expert_input = input("AI is uncertain. Please provide expert input: ")
            conversation_history.append(HumanMessage(content=expert_input))
            continue  # Re-run the loop with expert input

        # Add AI response to conversation history
        conversation_history.append(AIMessage(content=response.content))
        print("\nAI:", response.content)

if __name__ == "__main__":
    main()