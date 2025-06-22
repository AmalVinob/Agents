import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

class state(TypedDict):
    message: Annotated[list, add_messages]
    human_review: str
    final_response: str


def draft_response(state: state):
    response= llm.invoke(state["message"])
    print("\n Response:\n", response.content)
    review = input("approve or edit:\n ")
    return {"human_review": review}

def chatbot_response(state: state):
    try:
        prompt = f"""You are a helpful assistant. Based on this edited or approval(yes, no, etc) human input, generate a polished response:
        {state['human_review']} for the message {state['message'][-1]} and generate the summary more meaningfully and beautifully again once got response from human.
        """
        response = llm.invoke(prompt)
        return {"final_response": response.content}
    except Exception as e:
        print("Error during refinement:", e)
        return {"final_response": "Sorry, I couldn't generate a refined answer."}
    
graph_builder = StateGraph(state)
graph_builder.add_node("draft", draft_response)
graph_builder.add_node("chatbot_response", chatbot_response)

graph_builder.set_entry_point("draft")
graph_builder.add_edge("draft", "chatbot_response")
graph_builder.add_edge("chatbot_response", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    print("type 'Quit' or 'q' to exit")
    while True:
        user_input = input("you: ")
        if user_input.lower() in ["quit", "q"]:
            print("bye")
            break
        for event in graph.stream({"message": ("user", user_input)}):
            for value in event.values():
                final = value.get("final_response", "(No final response returned)")
                print("Assistant message: ", final)