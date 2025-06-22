import re
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os
import getpass
from typing import Literal
from langgraph.graph import StateGraph, MessagesState


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("GROQ_API_KEY")


@tool
def get_whether(location: str):
    """call to get current wheather."""
    if location.lower() in ['sf', 'san francisco']:
        return "its 60 degree and foggy"
    else:
        return "its 90 degree and sunny"
    
@tool
def get_coolest_cty():
    """get a list of coolest city."""
    return "nyc, sf"

tools = [
    get_whether,
    get_coolest_cty
]

tool_node = ToolNode(tools=tools)

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_whether",
            "args": {"location": "sf"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

output = tool_node.invoke({"messages": [message_with_single_tool_call]})
print(output)
# {'messages': [ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='tool_call_id')]}



