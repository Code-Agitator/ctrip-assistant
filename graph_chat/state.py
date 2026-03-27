from typing import List

from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
