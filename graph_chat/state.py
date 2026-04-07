from typing import List, Optional, Literal

from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """
    更新对话状态栈。
    参数:
        left (list[str]): 当前的状态栈。
        right (Optional[str]): 想要添加到栈中的新状态或动作。如果为 None，则不做任何更改；
                               如果为 "pop"，则弹出栈顶元素；否则将该值添加到栈中。
    返回:
        list[str]: 更新后的状态栈。
    """
    if right is None:
        return left  # 如果right是None，保持当前状态栈不变
    if right == "pop":
        return left[:-1]  # 如果right是"pop"，移除栈顶元素（即最后一个状态）
    return left + [right]  # 否则，将right添加到状态栈中


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            # 其元素严格限定为上述五个字符串值之一。这种做法确保了对话状态管理逻辑的一致性和正确性，避免了意外的状态值导致的潜在问题。
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]
