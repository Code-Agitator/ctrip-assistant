from langgraph.graph import StateGraph

from assistant import create_assistant_node
from graph_chat.state import State

builder = StateGraph(State)

builder.add_node('assistant', create_assistant_node())
